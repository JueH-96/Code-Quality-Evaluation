class Solution:
	def isGood(self, nums: List[int]) -> bool:
		n = max(nums)
		if len(nums) != n + 1:
			return False
		freq = {}
		for num in nums:
			freq[num] = freq.get(num, 0) + 1
		for i in range(1, n):
			if freq.get(i, 0) != 1:
				return False
		if freq.get(n, 0) != 2:
			return False
		return True