class Solution:
	def generateKey(self, num1: int, num2: int, num3: int) -> int:
		s1 = str(num1).zfill(4)
		s2 = str(num2).zfill(4)
		s3 = str(num3).zfill(4)
		key_str = ''.join(min(a, b, c) for a, b, c in zip(s1, s2, s3))
		return int(key_str)