from typing import List

class Solution:
	def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
		in_edges = [[] for _ in range(n)]
		ans = []
		for u, v in queries:
			in_edges[v].append(u)
			dist = [0] * n
			for i in range(1, n):
				dist[i] = dist[i-1] + 1
				for u_node in in_edges[i]:
					if dist[u_node] + 1 < dist[i]:
						dist[i] = dist[u_node] + 1
			ans.append(dist[n-1])
		return ans