import sys
sys.setrecursionlimit(1000000)

class Solution:
	def minCostGoodCaption(self, caption: str) -> str:
		n = len(caption)
		INF = 10**18
		dp = [[[INF] * 3 for _ in range(26)] for __ in range(n+1)]
		parent = [[[None] * 3 for _ in range(26)] for __ in range(n+1)]
		
		for c in range(26):
			char_c = chr(ord('a') + c)
			cost0 = abs(ord(char_c) - ord(caption[0]))
			dp[0][c][0] = cost0
		
		for i in range(0, n):
			for c in range(26):
				for k in range(3):
					if dp[i][c][k] == INF:
						continue
					current_char = chr(ord('a') + c)
					cost_extend = abs(ord(current_char) - ord(caption[i]))
					if k == 0:
						new_k = 1
					else:
						new_k = 2
					new_cost = dp[i][c][k] + cost_extend
					if new_cost < dp[i+1][c][new_k]:
						dp[i+1][c][new_k] = new_cost
						parent[i+1][c][new_k] = (c, k)
					
					if k == 2:
						for d in range(26):
							if d == c:
								continue
							char_d = chr(ord('a') + d)
							cost_break = abs(ord(char_d) - ord(caption[i]))
							new_cost_break = dp[i][c][k] + cost_break
							if new_cost_break < dp[i+1][d][0]:
								dp[i+1][d][0] = new_cost_break
								parent[i+1][d][0] = (c, k)
		
		min_ops = INF
		for c in range(26):
			if dp[n][c][2] < min_ops:
				min_ops = dp[n][c][2]
		
		if min_ops == INF:
			return ""
		
		candidates = []
		for c in range(26):
			if dp[n][c][2] == min_ops:
				s_list = [''] * n
				current_c = c
				current_k = 2
				for i in range(n-1, -1, -1):
					s_list[i] = chr(ord('a') + current_c)
					prev_state = parent[i+1][current_c][current_k]
					if prev_state is None:
						break
					prev_c, prev_k = prev_state
					current_c = prev_c
					current_k = prev_k
				candidate = ''.join(s_list)
				candidates.append(candidate)
		
		candidates.sort()
		return candidates[0]