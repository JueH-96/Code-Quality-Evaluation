class Solution:
    def findValidPair(self, s: str) -> str:
        freq = [0] * 10
        for char in s:
            digit = int(char)
            freq[digit] += 1
        
        for i in range(len(s) - 1):
            a = s[i]
            b = s[i+1]
            if a != b:
                d1 = int(a)
                d2 = int(b)
                if freq[d1] == d1 and freq[d2] == d2:
                    return a + b
        return ""