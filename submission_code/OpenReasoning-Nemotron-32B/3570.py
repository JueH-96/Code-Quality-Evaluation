class Solution:
	def countOfSubstrings(self, word: str, k: int) -> int:
		n = len(word)
		vowels = "aeiou"
		total = 0
		for start in range(n):
			seen = set()
			cons_count = 0
			for end in range(start, n):
				char = word[end]
				if char in vowels:
					seen.add(char)
				else:
					cons_count += 1
				if cons_count > k:
					break
				if cons_count == k and len(seen) == 5:
					total += 1
		return total