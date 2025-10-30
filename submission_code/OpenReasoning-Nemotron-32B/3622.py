from typing import List

class Solution:
	def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
		if not nums:
			return 0
		min_val = min(nums)
		max_val = max(nums)
		low_bound = min_val - k
		high_bound = max_val + k
		n_range = high_bound - low_bound + 1
		
		base_arr = [0] * n_range
		for a in nums:
			if low_bound <= a <= high_bound:
				idx = a - low_bound
				base_arr[idx] += 1
				
		diff = [0] * (n_range + 2)
		
		for a in nums:
			L = a - k
			R = a + k
			L_bound = max(low_bound, L)
			R_bound = min(high_bound, R)
			if L_bound <= R_bound:
				idxL = L_bound - low_bound
				idxR = R_bound - low_bound
				diff[idxL] += 1
				if idxR + 1 < len(diff):
					diff[idxR+1] -= 1
					
		count_arr = [0] * n_range
		current = 0
		for i in range(n_range):
			current += diff[i]
			count_arr[i] = current
			
		max_freq = 0
		for i in range(n_range):
			total_in_interval = count_arr[i]
			base_val = base_arr[i]
			candidate = min(total_in_interval, base_val + numOperations)
			if candidate > max_freq:
				max_freq = candidate
				
		return max_freq