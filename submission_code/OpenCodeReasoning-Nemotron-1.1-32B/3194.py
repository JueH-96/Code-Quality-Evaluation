class Solution:
	def findWordsContaining(self, words: List[str], x: str) -> List[int]:
		result = []
		for idx, word in enumerate(words):
			if x in word:
				result.append(idx)
		return result