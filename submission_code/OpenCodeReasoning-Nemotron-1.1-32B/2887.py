class Solution:
	def sortVowels(self, s: str) -> str:
		vowels = "aeiouAEIOU"
		indices = []
		vowel_chars = []
		for index, char in enumerate(s):
			if char in vowels:
				indices.append(index)
				vowel_chars.append(char)
		vowel_chars.sort()
		s_list = list(s)
		for idx, char in zip(indices, vowel_chars):
			s_list[idx] = char
		return ''.join(s_list)