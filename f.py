from weaviate.classes.query import MetadataQuery

jeopardy = client.collections.use("JeopardyQuestion")
response = jeopardy.query.hybrid(
    query="food",
    alpha=0.5,
    return_metadata=MetadataQuery(score=True, explain_score=True),
    limit=3,
)

for o in response.objects:
    print(o.properties)
    print(o.metadata.score, o.metadata.explain_score)