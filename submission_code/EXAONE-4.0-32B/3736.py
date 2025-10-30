class Solution:
    def findValidPair(self, s: str) -> str:
        from collections import defaultdict
        freq = defaultdict(int)
        for char in s:
            freq[char] += 1
        
        n = len(s)
        for i in range(n - 1):
            a, b = s[i], s[i+1]
            if a != b:
                if freq[a] == int(a) and freq[b] == int(b):
                    return s[i:i+2]
        return ""