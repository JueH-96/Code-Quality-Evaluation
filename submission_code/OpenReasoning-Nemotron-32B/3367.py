class Solution:
	def sumOfEncryptedInt(self, nums: List[int]) -> int:
		total = 0
		for num in nums:
			s = str(num)
			max_digit = max(s)
			encrypted_num = int(max_digit * len(s))
			total += encrypted_num
		return total