class Solution:
	def minimumTimeToInitialState(self, word: str, k: int) -> int:
		n = len(word)
		for T in range(1, n + 1):
			if T * k >= n:
				return T
			if word == word[T * k:] + word[:T * k]:
				return T
		return n