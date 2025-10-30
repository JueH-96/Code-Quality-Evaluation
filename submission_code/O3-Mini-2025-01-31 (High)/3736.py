class Solution:
    def findValidPair(self, s: str) -> str:
        from collections import Counter
        count = Counter(s)
        
        # Iterate over each adjacent pair in the string
        for i in range(len(s) - 1):
            d1, d2 = s[i], s[i+1]
            # The digits must be different
            if d1 != d2:
                # Check if each digit appears exactly as many times as its numeric value
                if count[d1] == int(d1) and count[d2] == int(d2):
                    return d1 + d2
        # Return empty string if no valid pair is found
        return ""