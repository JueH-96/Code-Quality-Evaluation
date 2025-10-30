class Solution:
	def maximumBeauty(self, nums: List[int], k: int) -> int:
		min_val = min(nums)
		max_val = max(nums)
		low_bound = min_val - k
		high_bound = max_val + k
		n_range = high_bound - low_bound + 1
		
		diff = [0] * (n_range + 1)
		
		for num in nums:
			a = num - k
			b = num + k
			idx_a = a - low_bound
			idx_b_plus = b - low_bound + 1
			
			diff[idx_a] += 1
			if idx_b_plus < len(diff):
				diff[idx_b_plus] -= 1
		
		max_coverage = 0
		current = 0
		for i in range(n_range):
			current += diff[i]
			if current > max_coverage:
				max_coverage = current
				
		return max_coverage