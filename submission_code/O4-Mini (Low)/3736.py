from collections import Counter

class Solution:
    def findValidPair(self, s: str) -> str:
        # Count how many times each digit appears in s
        cnt = Counter(s)
        
        # Iterate over adjacent pairs
        for i in range(len(s) - 1):
            a, b = s[i], s[i+1]
            # Check they are different and each appears exactly its numeric value times
            if a != b and cnt[a] == int(a) and cnt[b] == int(b):
                return a + b
        
        # No valid pair found
        return ""