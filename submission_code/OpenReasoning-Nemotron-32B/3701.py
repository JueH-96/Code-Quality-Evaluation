class Solution:
	def minCostGoodCaption(self, caption: str) -> str:
		n = len(caption)
		INF = 10**9
		dp = [[[INF] * 3 for _ in range(26)] for __ in range(n)]
		parent = [[[None] * 3 for _ in range(26)] for __ in range(n)]
		
		for c in range(26):
			cost0 = abs(ord(caption[0]) - (ord('a') + c))
			dp[0][c][0] = cost0
		
		for i in range(1, n):
			for c_prev in range(26):
				for l_prev in range(3):
					if dp[i-1][c_prev][l_prev] == INF:
						continue
					for c in range(26):
						cost_i = abs(ord(caption[i]) - (ord('a') + c))
						total_cost = dp[i-1][c_prev][l_prev] + cost_i
						if c == c_prev:
							new_l = l_prev + 1
							if new_l >= 3:
								new_l = 2
							if total_cost < dp[i][c][new_l]:
								dp[i][c][new_l] = total_cost
								parent[i][c][new_l] = (c_prev, l_prev)
						else:
							if l_prev == 2:
								new_l = 0
								if total_cost < dp[i][c][0]:
									dp[i][c][0] = total_cost
									parent[i][c][0] = (c_prev, l_prev)
		
		min_cost = INF
		for c in range(26):
			if dp[n-1][c][2] < min_cost:
				min_cost = dp[n-1][c][2]
		
		if min_cost == INF:
			return ""
		
		candidates = []
		for c in range(26):
			if dp[n-1][c][2] == min_cost:
				s_list = [''] * n
				cur_c = c
				cur_l = 2
				s_list[n-1] = chr(ord('a') + cur_c)
				for i in range(n-1, 0, -1):
					c_prev, l_prev = parent[i][cur_c][cur_l]
					s_list[i-1] = chr(ord('a') + c_prev)
					cur_c = c_prev
					cur_l = l_prev
				candidates.append(''.join(s_list))
		
		candidates.sort()
		return candidates[0]