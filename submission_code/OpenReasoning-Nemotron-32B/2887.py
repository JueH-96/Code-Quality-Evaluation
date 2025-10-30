class Solution:
	def sortVowels(self, s: str) -> str:
		vowels_set = set("aeiouAEIOU")
		vowel_list = sorted([char for char in s if char in vowels_set])
		vowel_iter = iter(vowel_list)
		result = []
		for char in s:
			if char in vowels_set:
				result.append(next(vowel_iter))
			else:
				result.append(char)
		return ''.join(result)