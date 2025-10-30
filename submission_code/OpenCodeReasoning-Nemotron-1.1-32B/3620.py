class Solution:
	def maxDistinctElements(self, nums: List[int], k: int) -> int:
		nums.sort()
		last = -10**18
		count = 0
		for x in nums:
			low = x - k
			high = x + k
			candidate = max(last + 1, low)
			if candidate <= high:
				count += 1
				last = candidate
		return count