class Solution:
	def maxScore(self, nums: List[int], x: int) -> int:
		even = -10**18
		odd = -10**18
		if nums[0] % 2 == 0:
			even = nums[0]
		else:
			odd = nums[0]
		
		for i in range(1, len(nums)):
			num = nums[i]
			if num % 2 == 0:
				candidate = max(even, odd - x) + num
				if candidate > even:
					even = candidate
			else:
				candidate = max(odd, even - x) + num
				if candidate > odd:
					odd = candidate
		
		return max(even, odd)