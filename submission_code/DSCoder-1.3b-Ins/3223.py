class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        count = 0
        for i in range(n):
            freq = [0]*26
            freq[ord(word[i]) - ord('a')] += 1
            for j in range(i+1, n):
                if freq[ord(word[j]) - ord('a')] < k:
                    break
                if j == n - 1:
                    count += 1
        return count