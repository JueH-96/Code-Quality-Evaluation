class Solution:
	def maximumStrongPairXor(self, nums: List[int]) -> int:
		n = len(nums)
		max_xor = 0
		for i in range(n):
			for j in range(i, n):
				x = nums[i]
				y = nums[j]
				if max(x, y) <= 2 * min(x, y):
					xor_val = x ^ y
					if xor_val > max_xor:
						max_xor = xor_val
		return max_xor