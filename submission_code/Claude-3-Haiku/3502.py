class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        count = 0
        n = len(s)
        
        for i in range(n):
            freq = [0] * 26
            for j in range(i, n):
                freq[ord(s[j]) - ord('a')] += 1
                if all(freq[i] >= k for i in range(26)):
                    count += n - j
                    break
        
        return count