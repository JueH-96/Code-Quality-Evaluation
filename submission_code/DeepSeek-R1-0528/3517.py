class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        end_edges = [[] for _ in range(n)]
        res = []
        
        for u, v in queries:
            end_edges[v].append(u)
            d = [0] * n
            d[0] = 0
            for i in range(1, n):
                d[i] = d[i-1] + 1
                for x in end_edges[i]:
                    if d[x] + 1 < d[i]:
                        d[i] = d[x] + 1
            res.append(d[n-1])
        
        return res