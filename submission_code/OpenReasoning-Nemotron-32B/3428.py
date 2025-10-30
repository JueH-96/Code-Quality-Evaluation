class Solution:
	def duplicateNumbersXOR(self, nums: List[int]) -> int:
		total_xor = 0
		distinct_xor = 0
		seen = set()
		for num in nums:
			total_xor ^= num
			if num not in seen:
				seen.add(num)
				distinct_xor ^= num
		return distinct_xor ^ total_xor