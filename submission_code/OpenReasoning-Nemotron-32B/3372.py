class Solution:
	def longestMonotonicSubarray(self, nums: List[int]) -> int:
		n = len(nums)
		if n <= 1:
			return n
		
		max_len = 1
		inc = 1
		dec = 1
		
		for i in range(1, n):
			if nums[i] > nums[i-1]:
				inc += 1
				dec = 1
			elif nums[i] < nums[i-1]:
				dec += 1
				inc = 1
			else:
				inc = 1
				dec = 1
			
			current_max = max(inc, dec)
			if current_max > max_len:
				max_len = current_max
		
		return max_len