class Solution:
    def findValidPair(self, s: str) -> str:
        # Create a dictionary to store the count of each digit
        digit_count = {}
        for digit in s:
            digit_count[digit] = digit_count.get(digit, 0) + 1
        
        # Iterate through the string and find the first valid pair
        for i in range(len(s) - 1):
            if s[i] != s[i+1] and digit_count[s[i]] == int(s[i]) and digit_count[s[i+1]] == int(s[i+1]):
                return s[i:i+2]
        
        # If no valid pair is found, return an empty string
        return ""