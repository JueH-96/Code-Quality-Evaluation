class Solution:
	def duplicateNumbersXOR(self, nums: List[int]) -> int:
		seen = set()
		result = 0
		for num in nums:
			if num in seen:
				result ^= num
			else:
				seen.add(num)
		return result