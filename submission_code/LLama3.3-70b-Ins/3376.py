from typing import List

class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        def longest_common_suffix(word1: str, word2: str) -> str:
            """Find the longest common suffix between two words."""
            i = len(word1) - 1
            j = len(word2) - 1
            common_suffix = ""
            while i >= 0 and j >= 0 and word1[i] == word2[j]:
                common_suffix = word1[i] + common_suffix
                i -= 1
                j -= 1
            return common_suffix

        result = []
        for query in wordsQuery:
            max_length = 0
            min_length = float('inf')
            index = -1
            for i, word in enumerate(wordsContainer):
                common_suffix = longest_common_suffix(word, query)
                if len(common_suffix) > max_length:
                    max_length = len(common_suffix)
                    min_length = len(word)
                    index = i
                elif len(common_suffix) == max_length:
                    if len(word) < min_length:
                        min_length = len(word)
                        index = i
            result.append(index)
        return result