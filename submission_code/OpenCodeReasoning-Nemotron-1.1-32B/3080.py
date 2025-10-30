class Solution:
	def maxSubarrays(self, nums: List[int]) -> int:
		total = nums[0]
		for i in range(1, len(nums)):
			total &= nums[i]
		
		if total != 0:
			return 1
		else:
			count = 0
			current = -1
			for num in nums:
				current &= num
				if current == 0:
					count += 1
					current = -1
			return count