class Solution:
	def countAlternatingSubarrays(self, nums: List[int]) -> int:
		total = 0
		cur = 0
		for i in range(len(nums)):
			if i > 0 and nums[i] != nums[i-1]:
				cur += 1
			else:
				cur = 1
			total += cur
		return total