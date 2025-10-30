class Solution:
	def findKOr(self, nums: List[int], k: int) -> int:
		result = 0
		for i in range(32):
			count = 0
			for num in nums:
				if (num >> i) & 1:
					count += 1
			if count >= k:
				result |= (1 << i)
		return result