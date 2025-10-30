class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        answer = []
        dist = list(range(n))
        for u, v in queries:
            for i in range(u + 1):
                dist[v] = min(dist[v], dist[i] + 1)
            for i in range(v + 1, n):
                dist[i] = min(dist[i], dist[i - 1] + 1)
            answer.append(dist[-1])
        return answer