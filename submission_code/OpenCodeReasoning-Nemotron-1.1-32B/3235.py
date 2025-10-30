class Solution:
	def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
		n_letters = 26
		INF = 10**18
		dist = [[INF] * n_letters for _ in range(n_letters)]
		
		for i in range(n_letters):
			dist[i][i] = 0
			
		for i in range(len(original)):
			u = ord(original[i]) - ord('a')
			v = ord(changed[i]) - ord('a')
			w = cost[i]
			if w < dist[u][v]:
				dist[u][v] = w
				
		for k in range(n_letters):
			for i in range(n_letters):
				if dist[i][k] == INF:
					continue
				for j in range(n_letters):
					if dist[k][j] != INF:
						if dist[i][j] > dist[i][k] + dist[k][j]:
							dist[i][j] = dist[i][k] + dist[k][j]
		
		total_cost = 0
		for i in range(len(source)):
			s_char = source[i]
			t_char = target[i]
			if s_char == t_char:
				continue
			u = ord(s_char) - ord('a')
			v = ord(t_char) - ord('a')
			if dist[u][v] == INF:
				return -1
			total_cost += dist[u][v]
			
		return total_cost