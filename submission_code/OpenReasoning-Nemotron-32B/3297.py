class Solution:
	def minimumTimeToInitialState(self, word: str, k: int) -> int:
		n = len(word)
		t = 1
		while True:
			if t * k >= n:
				return t
			if word[:n - t * k] == word[t * k:]:
				return t
			t += 1