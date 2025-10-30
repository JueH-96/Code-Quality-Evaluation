class Solution:
	def minOperations(self, nums: List[int], k: int) -> int:
		xor = 0
		for num in nums:
			xor ^= num
		diff = xor ^ k
		return bin(diff).count('1')