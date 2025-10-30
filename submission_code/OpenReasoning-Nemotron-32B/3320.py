class Solution:
	def maxOperations(self, nums: List[int]) -> int:
		s = nums[0] + nums[1]
		k = 1
		i = 2
		n = len(nums)
		while i < n - 1:
			if nums[i] + nums[i+1] == s:
				k += 1
				i += 2
			else:
				break
		return k