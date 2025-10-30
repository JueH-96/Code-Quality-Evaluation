import itertools
from typing import List

class Solution:
	def maximumValueSum(self, board: List[List[int]]) -> int:
		m = len(board)
		n = len(board[0])
		ans = -10**18
		
		masks_for_row0 = [mask for mask in range(8) if mask & 1 == 0]
		masks_for_row1 = [mask for mask in range(8) if mask & 2 == 0]
		masks_for_row2 = [mask for mask in range(8) if mask & 4 == 0]
		
		for rows in itertools.combinations(range(m), 3):
			rows_list = [board[i] for i in rows]
			dp = [-10**18] * 8
			dp[0] = 0
			
			for j in range(n):
				new_dp = dp.copy()
				
				a0 = rows_list[0][j]
				for mask in masks_for_row0:
					new_mask = mask | 1
					candidate = dp[mask] + a0
					if candidate > new_dp[new_mask]:
						new_dp[new_mask] = candidate
				
				a1 = rows_list[1][j]
				for mask in masks_for_row1:
					new_mask = mask | 2
					candidate = dp[mask] + a1
					if candidate > new_dp[new_mask]:
						new_dp[new_mask] = candidate
				
				a2 = rows_list[2][j]
				for mask in masks_for_row2:
					new_mask = mask | 4
					candidate = dp[mask] + a2
					if candidate > new_dp[new_mask]:
						new_dp[new_mask] = candidate
				
				dp = new_dp
			
			if dp[7] > ans:
				ans = dp[7]
				
		return ans