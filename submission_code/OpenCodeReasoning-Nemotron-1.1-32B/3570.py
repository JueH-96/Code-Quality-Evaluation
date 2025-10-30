class Solution:
	def countOfSubstrings(self, word: str, k: int) -> int:
		vowels_set = set('aeiou')
		n = len(word)
		count = 0
		for start in range(n):
			distinct_vowels = set()
			consonant_count = 0
			for end in range(start, n):
				char = word[end]
				if char in vowels_set:
					distinct_vowels.add(char)
				else:
					consonant_count += 1
				
				if len(distinct_vowels) == 5 and consonant_count == k:
					count += 1
				
				if len(distinct_vowels) == 5 and consonant_count > k:
					break
		return count