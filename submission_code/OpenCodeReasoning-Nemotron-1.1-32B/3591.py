class Solution:
	def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
		n = 26
		INF = 10**18
		dist = [[INF] * n for _ in range(n)]
		
		for i in range(n):
			dist[i][i] = 0
			j1 = (i + 1) % n
			dist[i][j1] = min(dist[i][j1], nextCost[i])
			j2 = (i - 1) % n
			dist[i][j2] = min(dist[i][j2], previousCost[i])
		
		for k in range(n):
			for i in range(n):
				if dist[i][k] == INF:
					continue
				for j in range(n):
					if dist[k][j] == INF:
						continue
					new_cost = dist[i][k] + dist[k][j]
					if new_cost < dist[i][j]:
						dist[i][j] = new_cost
		
		total_cost = 0
		for i in range(len(s)):
			a = ord(s[i]) - ord('a')
			b = ord(t[i]) - ord('a')
			total_cost += dist[a][b]
		
		return total_cost