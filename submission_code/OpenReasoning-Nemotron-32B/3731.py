class Solution:
	def subarraySum(self, nums: List[int]) -> int:
		n = len(nums)
		prefix = [0] * (n + 1)
		for i in range(1, n + 1):
			prefix[i] = prefix[i - 1] + nums[i - 1]
		
		total = 0
		for i in range(n):
			start_index = max(0, i - nums[i])
			total += prefix[i + 1] - prefix[start_index]
		return total