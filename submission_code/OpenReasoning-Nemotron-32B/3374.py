class Solution:
	def countAlternatingSubarrays(self, nums: List[int]) -> int:
		count = 0
		current = 0
		for i in range(len(nums)):
			if i > 0 and nums[i] == nums[i-1]:
				current = 1
			else:
				current += 1
			count += current
		return count