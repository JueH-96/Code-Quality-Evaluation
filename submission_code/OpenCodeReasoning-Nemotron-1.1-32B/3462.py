class Solution:
	def doesAliceWin(self, s: str) -> bool:
		vowels = set('aeiou')
		for char in s:
			if char in vowels:
				return True
		return False