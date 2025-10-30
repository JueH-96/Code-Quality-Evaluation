import math
from typing import List

class Solution:
	def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
		distinct_target = set(target)
		if not distinct_target:
			return 0
		sorted_target = sorted(distinct_target)
		reduced_target = []
		for i in range(len(sorted_target)-1, -1, -1):
			t = sorted_target[i]
			found = False
			for x in reduced_target:
				if x % t == 0:
					found = True
					break
			if not found:
				reduced_target.append(t)
		m = len(reduced_target)
		if m == 0:
			return 0
		
		lcm_arr = [0] * (1 << m)
		for mask in range(1, 1 << m):
			lcm_val = 1
			for j in range(m):
				if mask & (1 << j):
					t_val = reduced_target[j]
					g = math.gcd(lcm_val, t_val)
					lcm_val = (lcm_val * t_val) // g
			lcm_arr[mask] = lcm_val
		
		dp = [10**18] * (1 << m)
		dp[0] = 0
		
		for x in nums:
			costs = [0] * (1 << m)
			for mask in range(1, 1 << m):
				L = lcm_arr[mask]
				if x < L:
					cost_val = L - x
				else:
					remainder = x % L
					if remainder == 0:
						cost_val = 0
					else:
						cost_val = L - remainder
				costs[mask] = cost_val
			
			new_dp = dp.copy()
			for mask in range(1 << m):
				if dp[mask] == 10**18:
					continue
				for s in range(1, 1 << m):
					new_mask = mask | s
					total_cost = dp[mask] + costs[s]
					if total_cost < new_dp[new_mask]:
						new_dp[new_mask] = total_cost
			dp = new_dp
		
		return dp[(1 << m) - 1]