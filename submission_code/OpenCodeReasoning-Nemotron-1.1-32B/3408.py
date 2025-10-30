class Solution:
	def numberOfSpecialChars(self, word: str) -> int:
		word_set = set(word)
		base_set = set(word.lower())
		return sum(1 for base in base_set if base in word_set and base.upper() in word_set)