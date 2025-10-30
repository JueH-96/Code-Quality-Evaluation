class Solution:
	def numberOfSpecialChars(self, word: str) -> int:
		has_lower = [False] * 26
		has_upper = [False] * 26
		for char in word:
			if char.islower():
				idx = ord(char) - ord('a')
				has_lower[idx] = True
			elif char.isupper():
				idx = ord(char) - ord('A')
				has_upper[idx] = True
		count = 0
		for i in range(26):
			if has_lower[i] and has_upper[i]:
				count += 1
		return count