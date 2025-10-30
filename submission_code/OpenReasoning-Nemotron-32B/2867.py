class Solution:
	def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
		mod = 10**9 + 7
		ones = [i for i, num in enumerate(nums) if num == 1]
		if not ones:
			return 0
		total = 1
		for i in range(len(ones) - 1):
			gap = ones[i+1] - ones[i] - 1
			total = (total * (gap + 1)) % mod
		return total