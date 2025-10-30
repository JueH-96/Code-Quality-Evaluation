class Solution:
	def sumOfEncryptedInt(self, nums: List[int]) -> int:
		total = 0
		for x in nums:
			s = str(x)
			max_digit = max(s)
			encrypted_str = max_digit * len(s)
			total += int(encrypted_str)
		return total