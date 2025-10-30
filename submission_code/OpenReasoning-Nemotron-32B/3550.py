class Solution:
	def maximumValueSum(self, board: List[List[int]]) -> int:
		m = len(board)
		n = len(board[0])
		dp = {(): 0}
		
		for i in range(m):
			next_dp = dp.copy()
			for state, total_val in dp.items():
				k = len(state)
				if k == 3:
					continue
				for j in range(n):
					if j in state:
						continue
					new_state_list = list(state) + [j]
					new_state_list.sort()
					new_state = tuple(new_state_list)
					new_total = total_val + board[i][j]
					if new_state in next_dp:
						if new_total > next_dp[new_state]:
							next_dp[new_state] = new_total
					else:
						next_dp[new_state] = new_total
			dp = next_dp
		
		ans = -10**18
		for state, total_val in dp.items():
			if len(state) == 3:
				if total_val > ans:
					ans = total_val
		return ans