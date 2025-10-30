from collections import defaultdict

class Solution:
	def maxPalindromesAfterOperations(self, words: List[str]) -> int:
		freq = defaultdict(int)
		for word in words:
			for c in word:
				freq[c] += 1
		
		total_odd = 0
		for count in freq.values():
			if count % 2 == 1:
				total_odd += 1
		
		odd_words = 0
		for word in words:
			if len(word) % 2 == 1:
				odd_words += 1
				
		cannot_make = max(0, (total_odd - odd_words + 1) // 2)
		return len(words) - cannot_make