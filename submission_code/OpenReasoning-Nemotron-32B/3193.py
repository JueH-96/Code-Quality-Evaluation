class Solution:
	def maximumStrongPairXor(self, nums: List[int]) -> int:
		max_xor = 0
		n = len(nums)
		for i in range(n):
			for j in range(i, n):
				a = min(nums[i], nums[j])
				b = max(nums[i], nums[j])
				if b <= 2 * a:
					xor_val = nums[i] ^ nums[j]
					if xor_val > max_xor:
						max_xor = xor_val
		return max_xor