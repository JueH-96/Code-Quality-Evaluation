class Solution:
    def findValidPair(self, s: str) -> str:
        # Compute frequency of each digit
        freq = [0] * 10
        for char in s:
            digit = int(char)
            freq[digit] += 1
        
        # Iterate through adjacent pairs
        for i in range(len(s) - 1):
            dig1 = int(s[i])
            dig2 = int(s[i + 1])
            if dig1 != dig2 and freq[dig1] == dig1 and freq[dig2] == dig2:
                return s[i:i + 2]
        
        # No valid pair found
        return ""