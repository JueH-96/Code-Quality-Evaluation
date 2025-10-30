class Solution:
	def maxDistinctElements(self, nums: list, k: int) -> int:
		nums.sort()
		last = -10**18
		count = 0
		for num in nums:
			low_bound = num - k
			high_bound = num + k
			candidate = max(last + 1, low_bound)
			if candidate <= high_bound:
				count += 1
				last = candidate
		return count