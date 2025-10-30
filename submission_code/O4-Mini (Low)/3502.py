class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        result = 0
        
        for i in range(n):
            freq = [0] * 26
            for j in range(i, n):
                idx = ord(s[j]) - ord('a')
                freq[idx] += 1
                # once any character reaches k occurrences, all longer substrings
                # starting at i with end >= j are valid
                if freq[idx] == k:
                    result += (n - j)
                    break
        
        return result