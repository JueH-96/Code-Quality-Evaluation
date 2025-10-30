class Solution:
	def maxScore(self, nums: List[int], x: int) -> int:
		n = len(nums)
		even_max = -10**18
		odd_max = -10**18
		
		if nums[0] % 2 == 0:
			even_max = nums[0]
		else:
			odd_max = nums[0]
			
		for i in range(1, n):
			if nums[i] % 2 == 0:
				current_val = max(even_max, odd_max - x) + nums[i]
				even_max = max(even_max, current_val)
			else:
				current_val = max(odd_max, even_max - x) + nums[i]
				odd_max = max(odd_max, current_val)
				
		return max(even_max, odd_max)