class Solution:
	def maxFrequency(self, nums: List[int], k: int) -> int:
		base_freq = nums.count(k)
		max_gain = -10**18
		
		for a in range(1, 51):
			if a == k:
				continue
			current_sum = 0
			best_sum = -10**18
			for num in nums:
				if num == a:
					x = 1
				elif num == k:
					x = -1
				else:
					x = 0
				current_sum = max(x, current_sum + x)
				best_sum = max(best_sum, current_sum)
			if best_sum > max_gain:
				max_gain = best_sum
		
		total = base_freq + max(0, max_gain)
		return total