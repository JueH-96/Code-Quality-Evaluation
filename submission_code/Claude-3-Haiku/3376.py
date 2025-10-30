class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        ans = []
        for query in wordsQuery:
            max_len = 0
            min_len = float('inf')
            min_idx = -1
            for i, word in enumerate(wordsContainer):
                suffix = self.longest_common_suffix(word, query)
                if len(suffix) > max_len or (len(suffix) == max_len and len(word) < min_len):
                    max_len = len(suffix)
                    min_len = len(word)
                    min_idx = i
            ans.append(min_idx)
        return ans
    
    def longest_common_suffix(self, word1: str, word2: str) -> str:
        i = 1
        while i <= min(len(word1), len(word2)) and word1[-i] == word2[-i]:
            i += 1
        return word1[-i+1:]