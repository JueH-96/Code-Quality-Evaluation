class Solution:
	def isValid(self, word: str) -> bool:
		if len(word) < 3:
			return False
		vowels = "aeiouAEIOU"
		has_vowel = False
		has_consonant = False
		for char in word:
			if char.isdigit():
				continue
			if char.isalpha():
				if char in vowels:
					has_vowel = True
				else:
					has_consonant = True
			else:
				return False
		return has_vowel and has_consonant