class Solution:
	def hasSameDigits(self, s: str) -> bool:
		while len(s) > 2:
			s = ''.join(str((int(a) + int(b)) % 10) for a, b in zip(s, s[1:]))
		return s[0] == s[1]