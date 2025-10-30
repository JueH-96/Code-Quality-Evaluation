class Solution:
    def stringIndices(self, wordsContainer: list[str], wordsQuery: list[str]) -> list[int]:
        def longest_common_suffix_length(s1, s2):
            i = 0
            while i < len(s1) and i < len(s2) and s1[-(i + 1)] == s2[-(i + 1)]:
                i += 1
            return i
        
        ans = []
        for query in wordsQuery:
            max_suffix_len = -1
            min_length = float('inf')
            chosen_index = -1
            for i, word in enumerate(wordsContainer):
                suffix_len = longest_common_suffix_length(word, query)
                if suffix_len > max_suffix_len or (suffix_len == max_suffix_len and len(word) < min_length):
                    max_suffix_len = suffix_len
                    min_length = len(word)
                    chosen_index = i
            ans.append(chosen_index)
        return ans