import time
import pyterrier as pt


def process_queries(query_df, index_ref):
    """
    This function processes the queries in the queries DataFrame following the
    PyTerrier Index configuration and retrieves each query term's postings in
    the index.

    Parameters:
    -----------
        query_df : pd.DataFrame
            The queries DataFrame. Assumes columns 'qid' and 'query'.

        index_ref : str
            The path to the PyTerrier IndexRef.

    Returns:
    --------
        average_query_processing_time : float
            Average time of processing a single query, in milliseconds.

        average_query_postings_time : float
            Average time of retrieving query postings, in milliseconds.
    """

    processing_time = 0
    postings_time = 0

    index = pt.IndexFactory.of(index_ref)
    lex = index.getLexicon()
    inv = index.getInvertedIndex()

    for row in pt.tqdm(query_df.itertuples(index=False), total=len(query_df)):
        start_time = time.time_ns()

        # We are working with Terrier-core
        rq = pt.terrier.J.Request()
        rq.setQueryID(row.qid)
        rq.setOriginalQuery(row.query)
        rq.setIndex(index)
        pt.terrier.J.TerrierQLParser().process(None, rq)
        pt.terrier.J.TerrierQLToMatchingQueryTerms().process(None, rq)
        pt.terrier.J.ApplyTermPipeline().process(None, rq)

        processing_time += time.time_ns() - start_time

        for term in rq.getMatchingQueryTerms():
            term_str = term.getKey().toString()
            le = lex.getLexiconEntry(term_str)

            if le is None:
                continue

            postings = inv.getPostings(le)

            for posting in postings:
                posting.getId()

        postings_time += time.time_ns() - start_time
    avg_processing_time = round(processing_time / 1_000_000 / len(query_df), 2)
    avg_postings_time = round(postings_time / 1_000_000 / len(query_df), 2)
    return avg_processing_time, avg_postings_time
