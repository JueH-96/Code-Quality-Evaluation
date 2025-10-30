class Solution:
	def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
		INF = 10**18
		n = len(source)
		min_cost = [[INF] * 26 for _ in range(26)]
		
		for i in range(26):
			min_cost[i][i] = 0
			
		for i in range(len(original)):
			orig_char = original[i]
			chan_char = changed[i]
			c = cost[i]
			idx1 = ord(orig_char) - ord('a')
			idx2 = ord(chan_char) - ord('a')
			if c < min_cost[idx1][idx2]:
				min_cost[idx1][idx2] = c
				
		for k in range(26):
			for i in range(26):
				if min_cost[i][k] == INF:
					continue
				for j in range(26):
					if min_cost[i][k] + min_cost[k][j] < min_cost[i][j]:
						min_cost[i][j] = min_cost[i][k] + min_cost[k][j]
						
		total_cost = 0
		for i in range(n):
			s_char = source[i]
			t_char = target[i]
			if s_char == t_char:
				continue
			s_idx = ord(s_char) - ord('a')
			t_idx = ord(t_char) - ord('a')
			if min_cost[s_idx][t_idx] == INF:
				return -1
			total_cost += min_cost[s_idx][t_idx]
			
		return total_cost