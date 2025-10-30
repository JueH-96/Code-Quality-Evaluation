class Solution:
    def findWordsContaining(self, words: List[str], x: str) -> List[int]:
        result = []
        for index, word in enumerate(words):
            if x in word:
                result.append(index)
        return result