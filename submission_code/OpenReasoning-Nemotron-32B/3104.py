class Solution:
	def countWays(self, nums: List[int]) -> int:
		n = len(nums)
		freq = [0] * (n + 1)
		for num in nums:
			if num < n:
				freq[num] += 1
		total = 0
		count = 0
		for k in range(n + 1):
			if freq[k] == 0:
				if total == k:
					count += 1
			total += freq[k]
		return count