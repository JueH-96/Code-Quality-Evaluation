class Solution:
    def stringIndices(self, wordsContainer, wordsQuery):
        def longest_common_suffix(s1, s2):
            i = len(s1) - 1
            j = len(s2) - 1
            while i >= 0 and j >= 0 and s1[i] == s2[j]:
                i -= 1
                j -= 1
            return len(s1) - i - 1

        ans = []
        for query in wordsQuery:
            max_suffix_len = -1
            min_len = float('inf')
            index = -1
            for i, word in enumerate(wordsContainer):
                suffix_len = longest_common_suffix(query, word)
                if suffix_len > max_suffix_len or (suffix_len == max_suffix_len and len(word) < min_len):
                    max_suffix_len = suffix_len
                    min_len = len(word)
                    index = i
            ans.append(index)
        return ans