class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        count = 0
        for i in range(len(word)):
            freq = [0]*26
            distinct = 0
            for j in range(i, len(word)):
                if freq[ord(word[j]) - ord('a')] == 0:
                    distinct += 1
                freq[ord(word[j]) - ord('a')] += 1
                if distinct * k == j - i + 1:
                    count += 1
        return count