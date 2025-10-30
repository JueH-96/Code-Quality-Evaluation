class Solution:
    def findValidPair(self, s: str) -> str:
        from collections import Counter
        counts = Counter(s)
        
        # Traverse adjacent pairs
        for i in range(len(s) - 1):
            a, b = s[i], s[i+1]
            # Digits must differ and each digit's total count must match its numeric value
            if a != b and counts[a] == int(a) and counts[b] == int(b):
                return a + b
        
        return ""