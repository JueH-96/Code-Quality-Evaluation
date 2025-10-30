from typing import List

class Solution:
	def maxPalindromesAfterOperations(self, words: List[str]) -> int:
		total_chars = 0
		freq = [0] * 26
		for word in words:
			total_chars += len(word)
			for c in word:
				idx = ord(c) - ord('a')
				freq[idx] += 1
		
		total_pairs = 0
		for count in freq:
			total_pairs += count // 2
		
		arr = [len(word) // 2 for word in words]
		arr.sort()
		s = 0
		count_pal = 0
		for a in arr:
			if s + a <= total_pairs:
				s += a
				count_pal += 1
			else:
				break
		return count_pal