class Solution:
    def findWordsContaining(self, words: List[str], x: str) -> List[int]:
        result_indices = []
        for i in range(len(words)):
            if x in words[i]:
                result_indices.append(i)
        return result_indices