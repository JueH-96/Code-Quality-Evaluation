from typing import List

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # Build initial incoming-edge lists for the path 0->1->2->...->n-1
        in_edges = [[] for _ in range(n)]
        for i in range(n-1):
            in_edges[i+1].append(i)
        
        ans = []
        INF = 10**9
        
        for u, v in queries:
            # Add the new edge u->v
            in_edges[v].append(u)
            
            # Compute shortest distances dp[i] = min number of edges from 0 to i
            dp = [INF]*n
            dp[0] = 0
            # Since all edges go from lower to higher index, 0..n-1 is a topological order
            for i in range(1, n):
                # Relax all incoming edges x->i
                best = INF
                for x in in_edges[i]:
                    if dp[x] + 1 < best:
                        best = dp[x] + 1
                dp[i] = best
            
            ans.append(dp[n-1])
        
        return ans