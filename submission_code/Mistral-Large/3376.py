from typing import List

class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        def longest_common_suffix(s1, s2):
            i, j = len(s1) - 1, len(s2) - 1
            while i >= 0 and j >= 0 and s1[i] == s2[j]:
                i -= 1
                j -= 1
            return s1[i+1:]

        def find_best_match(query):
            best_match = None
            best_length = float('inf')
            best_index = -1
            longest_suffix = ""

            for idx, word in enumerate(wordsContainer):
                common_suffix = longest_common_suffix(word, query)
                if len(common_suffix) > len(longest_suffix):
                    longest_suffix = common_suffix
                    best_match = word
                    best_length = len(word)
                    best_index = idx
                elif len(common_suffix) == len(longest_suffix):
                    if len(word) < best_length:
                        best_match = word
                        best_length = len(word)
                        best_index = idx
                    elif len(word) == best_length:
                        if idx < best_index:
                            best_match = word
                            best_index = idx

            return best_index

        result = []
        for query in wordsQuery:
            result.append(find_best_match(query))

        return result