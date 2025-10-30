class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        count = 0
        n = len(word)
        
        for i in range(n):
            freq = [0] * 26
            for j in range(i, n):
                freq[ord(word[j]) - ord('a')] += 1
                if all(f == k for f in freq) and all(abs(freq[i] - freq[i+1]) <= 2 for i in range(25)):
                    count += 1
                else:
                    break
        
        return count