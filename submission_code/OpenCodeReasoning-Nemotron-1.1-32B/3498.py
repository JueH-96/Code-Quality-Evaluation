from typing import List

class Solution:
	def minChanges(self, nums: List[int], k: int) -> int:
		n = len(nums)
		m = n // 2
		freq_d = [0] * (k + 1)
		freq_T = [0] * (k + 1)
		
		for i in range(m):
			a = nums[i]
			b = nums[n - 1 - i]
			d_val = abs(a - b)
			T_val = max(a, b, k - a, k - b)
			
			if d_val <= k:
				freq_d[d_val] += 1
			if T_val <= k:
				freq_T[T_val] += 1
				
		suffix_T = [0] * (k + 2)
		for x in range(k, -1, -1):
			suffix_T[x] = freq_T[x] + suffix_T[x + 1]
			
		best = float('inf')
		for X in range(0, k + 1):
			cost_here = n - suffix_T[X] - freq_d[X]
			if cost_here < best:
				best = cost_here
				
		return best