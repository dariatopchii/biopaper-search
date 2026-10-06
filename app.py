import json
import pandas as pd
import streamlit as st
from biopaper.data import ROOT, load_jsonl, load_papers, write_jsonl
from biopaper.retrieval import SearchEngine
from biopaper.evaluation import require_complete_pool
from scripts.evaluate import evaluate

st.set_page_config(page_title='BioPaper Search', page_icon='🧬', layout='wide')
st.title('BioPaper Search')
st.write('Find biomedical research by meaning. Compare retrieval methods and measure what improves.')


@st.cache_resource
def get_engine(signature):
    return SearchEngine(load_papers(signature[0]))


corpus = ROOT / 'data/papers.jsonl'
if not corpus.exists():
    st.info('Download the article collection first: python scripts/fetch_papers.py --limit 300')
    st.stop()
engine = get_engine((str(corpus), corpus.stat().st_mtime_ns, corpus.stat().st_size))
queries = json.loads((ROOT / 'evaluation/queries.json').read_text(encoding='utf-8'))
st.caption(f'{len(engine.papers)} abstracts · Europe PMC · English queries · literature discovery, not clinical advice')
search_tab, judge_tab, report_tab, progress_tab = st.tabs(['Search and compare', 'Relevance review', 'Experiment results', 'Project progress'])


def show_paper(paper):
    st.markdown(f"**{paper['rank']}. {paper['title']}**")
    st.caption(f"{paper['year']} · {paper['authors']}")
    st.write(paper['abstract'])
    st.link_button('Read source article', paper['url'])


with search_tab:
    example = st.selectbox('Try a research question', [q['text'] for q in queries[:10]])
    with st.form('search'):
        query = st.text_input('Your question', value=example, max_chars=1000)
        compare = st.checkbox('Compare all three methods', value=True)
        method = st.selectbox('Single method', ['bm25', 'dense', 'rerank'])
        k = st.slider('Results per method', 3, 10, 5)
        submit = st.form_submit_button('Find papers', type='primary')
    if submit:
        results = {}
        for name in (['bm25', 'dense', 'rerank'] if compare else [method]):
            try:
                with st.spinner('Finding papers. First use may download a model and build its index.'):
                    papers, elapsed = engine.search(query, name, k)
                results[name] = (papers, elapsed)
            except (ValueError, OSError, ImportError, RuntimeError) as e:
                st.error(f'{name}: {e}')
        st.session_state['results'] = results
    results = st.session_state.get('results', {})
    if results:
        for column, (name, (papers, elapsed)) in zip(st.columns(len(results)), results.items()):
            with column:
                st.subheader({'bm25': 'Keyword search', 'dense': 'Semantic search', 'rerank': 'Semantic + reranking'}[name])
                st.caption(f'{elapsed:.0f} ms · first-run timings include setup')
                if not papers:
                    st.info('No matching keywords. Try different words or semantic search.')
                for paper in papers:
                    with st.expander(f"{paper['rank']}. {paper['title']}", expanded=paper['rank'] == 1):
                        show_paper(paper)
        st.caption('Ranking scores use different scales and are not probabilities. Compare ranks and relevance, not raw scores.')

with judge_tab:
    st.write('Rate papers without seeing which search method retrieved them. Judge whether the abstract answers the research question, not just whether it shares keywords.')
    pool = [r for r in load_jsonl(ROOT / 'evaluation/pool.jsonl') if r['corpus_hash'] == engine.fingerprint]
    if not pool:
        st.info('Create the review pool first: python scripts/evaluate.py pool')
    else:
        chosen = st.selectbox('Research question', queries, format_func=lambda q: q['text'], key='review_q')
        judgments = load_jsonl(ROOT / 'evaluation/judgments.jsonl')
        lookup = {(r['query_id'], r['paper_id']): r for r in judgments if r['corpus_hash'] == engine.fingerprint}
        total_done = sum((r['query_id'], r['paper_id']) in lookup for r in pool)
        st.progress(total_done / len(pool), text=f'{total_done}/{len(pool)} reviewed across the whole experiment')
        docs = {p['id']: p for p in engine.papers}
        rows = [r for r in pool if r['query_id'] == chosen['id']]
        count = sum((r['query_id'], r['paper_id']) in lookup for r in rows)
        st.progress(count / max(len(rows), 1), text=f'{count}/{len(rows)} reviewed for this question')
        pending_only = st.checkbox('Show only unreviewed papers', value=True)
        visible = [r for r in rows if (r['query_id'], r['paper_id']) not in lookup] if pending_only else rows
        if not visible:
            st.success('This question is fully reviewed. Choose another question or uncheck the filter to revise ratings.')
        else:
            selected = st.selectbox('Paper to review', visible, format_func=lambda r: ('✓ ' if (r['query_id'], r['paper_id']) in lookup else '') + docs[r['paper_id']]['title'])
            paper = docs[selected['paper_id']]
            show_paper({**paper, 'rank': 1})
            current = lookup.get((chosen['id'], paper['id']), {}).get('grade')
            with st.form('grade'):
                grade = st.radio('Relevance', [0, 1, 2], index=current, key=f"grade_{chosen['id']}_{paper['id']}", format_func=lambda g: ['0 — irrelevant: does not address the question', '1 — partly useful: background or adjacent evidence', '2 — directly relevant: addresses the requested task'][g])
                note = st.text_input('Why did you choose this rating? (optional)', key=f"note_{chosen['id']}_{paper['id']}")
                save = st.form_submit_button('Save and continue')
            if save:
                if grade is None:
                    st.warning('Choose a relevance rating before saving.')
                else:
                    # Single-user local app: upsert by query, paper, and corpus snapshot.
                    judgments = [r for r in judgments if not (r['query_id'] == chosen['id'] and r['paper_id'] == paper['id'] and r['corpus_hash'] == engine.fingerprint)]
                    judgments.append({'query_id': chosen['id'], 'paper_id': paper['id'], 'grade': grade, 'note': note, 'corpus_hash': engine.fingerprint})
                    write_jsonl(ROOT / 'evaluation/judgments.jsonl', judgments)
                    st.rerun()

with report_tab:
    st.write('No quality scores are generated until you provide human relevance judgments.')
    split_choice = st.selectbox('Evaluation set', ['dev', 'test'], format_func=lambda s: 'Development — use this first' if s == 'dev' else 'Held-out test — use after settings are frozen')
    if split_choice == 'test':
        st.info('Use the held-out test only after deciding your models and search settings. Repeated tuning on test results makes the final comparison less reliable.')
    selected_queries = [q for q in queries if q['split'] == split_choice]
    saved = load_jsonl(ROOT / 'evaluation/judgments.jsonl')
    grades = {}
    for r in saved:
        if r['corpus_hash'] == engine.fingerprint:
            grades.setdefault(r['query_id'], {})[r['paper_id']] = r['grade']
    readiness_error = None
    try:
        require_complete_pool(selected_queries, load_jsonl(ROOT / 'evaluation/pool.jsonl'), grades, engine.fingerprint)
        if any(not any(g > 0 for g in grades[q['id']].values()) for q in selected_queries):
            readiness_error = 'At least one question has no relevant articles in its pool. Review the question and corpus coverage before evaluating.'
    except ValueError as e:
        readiness_error = str(e)
    if readiness_error:
        st.caption(readiness_error)
    if st.button('Run evaluation', disabled=readiness_error is not None, type='primary'):
        try:
            with st.spinner('Comparing relevance and measuring warmed search latency…'):
                evaluate(engine, split_choice, 5, 30)
            st.success('Evaluation completed. Results are shown below.')
        except (ValueError, OSError, ImportError, RuntimeError) as e:
            st.error(str(e))
    for split in ['dev', 'test']:
        path = ROOT / 'evaluation/reports' / (split + '.json')
        if path.exists():
            report = json.loads(path.read_text(encoding='utf-8'))
            if report['corpus_hash'] != engine.fingerprint:
                st.warning(f'{split}: report belongs to an older corpus. Run evaluation again.')
                continue
            st.subheader(f"{split.title()} · {report['query_count']} questions · top {report['k']}")
            st.dataframe(pd.DataFrame(report['summary']), hide_index=True)
            st.caption(report['limitations'])
            st.download_button('Download experiment report', path.read_text(encoding='utf-8'), file_name=path.name, key=split)
    st.caption('Pooled recall uses relevant documents found in the review pool, not all relevant articles in the corpus.')

    st.subheader('Learning from preferences')
    st.write('Can a small trained ranker improve relevance? It learns article preferences induced from your explicit ratings: 2 > 1 > 0. Equal ratings create no preference.')
    st.caption('Fixed split: first 15 development questions train; last 5 validate; 10 test questions remain held out. Features: keyword score, semantic similarity, and title overlap.')
    preference_error = None
    try:
        require_complete_pool([q for q in queries if q['split'] == 'dev'], load_jsonl(ROOT / 'evaluation/pool.jsonl'), grades, engine.fingerprint)
    except ValueError as e:
        preference_error = str(e)
    if preference_error:
        st.caption(preference_error)
    if st.button('Run preference learning experiment', disabled=preference_error is not None):
        try:
            from scripts.preference_experiment import experiment
            with st.spinner('Training and evaluating on separate validation questions…'):
                experiment(engine)
            st.success('Preference experiment completed.')
        except (ValueError, OSError, ImportError, RuntimeError) as e:
            st.warning(str(e))
    curve_path = ROOT / 'evaluation/reports/preferences_validation.json'
    if curve_path.exists():
        learned = json.loads(curve_path.read_text(encoding='utf-8'))
        from biopaper.preferences import annotation_snapshot
        if learned['corpus_hash'] == engine.fingerprint and learned.get('annotation_snapshot') == annotation_snapshot(queries, saved, engine.fingerprint):
            curve = pd.DataFrame(learned['curves'])
            st.line_chart(curve.pivot(index='training_queries', columns='seed', values='mean_ndcg_at_5'), x_label='Training questions', y_label='Validation nDCG@5')
            st.dataframe(curve, hide_index=True)
            st.caption(learned['limitations'])
        else:
            st.info('Preference results belong to older data, questions, or ratings. Rerun the experiment to update them.')

with progress_tab:
    st.subheader('The question we are working towards')
    st.write('Can a lightweight ranker trained on a small number of human preferences improve biomedical literature search?')
    st.info('Software is working. Human annotations and measured quality improvements are still pending.')
    for filename, title in [('ROADMAP.md', 'Milestones'), ('PROGRESS.md', 'Evidence and next step'), ('EXPERIMENT.md', 'Experiment protocol')]:
        document = ROOT / 'docs' / filename
        if document.exists():
            with st.expander(title, expanded=filename == 'ROADMAP.md'):
                st.markdown(document.read_text(encoding='utf-8'))
