class Solution:
	def doesAliceWin(self, s: str) -> bool:
		vowels = "aeiou"
		cnt = 0
		for c in s:
			if c in vowels:
				cnt += 1
		if cnt == 0:
			return False
		if cnt % 2 == 1:
			return True
		for i in range(len(s) - 1):
			if s[i] == s[i+1]:
				return True
		return False