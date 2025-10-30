from collections import deque

class Solution:
	def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
		graph = [[] for _ in range(n)]
		for i in range(n-1):
			graph[i].append(i+1)
		
		dp = [0] * n
		for i in range(1, n):
			dp[i] = dp[i-1] + 1
		
		ans = []
		for u, v in queries:
			graph[u].append(v)
			if dp[u] + 1 < dp[v]:
				dp[v] = dp[u] + 1
				q = deque([v])
				while q:
					node = q.popleft()
					for neighbor in graph[node]:
						if dp[node] + 1 < dp[neighbor]:
							dp[neighbor] = dp[node] + 1
							q.append(neighbor)
			ans.append(dp[n-1])
		
		return ans