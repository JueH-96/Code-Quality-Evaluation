class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        def longest_common_suffix(s1, s2):
            min_len = min(len(s1), len(s2))
            for i in range(1, min_len + 1):
                if s1[-i] != s2[-i]:
                    return i - 1
            return min_len

        ans = []
        for query in wordsQuery:
            max_suffix_len = 0
            best_index = -1
            for i, word in enumerate(wordsContainer):
                suffix_len = longest_common_suffix(word, query)
                if (suffix_len > max_suffix_len or
                    (suffix_len == max_suffix_len and (best_index == -1 or len(word) < len(wordsContainer[best_index])))):
                    max_suffix_len = suffix_len
                    best_index = i
            ans.append(best_index)

        return ans