class Solution:
    def findValidPair(self, s: str) -> str:
        # Count occurrences of each digit
        count = {}
        for digit in s:
            count[digit] = count.get(digit, 0) + 1
        
        # Traverse the string to find valid pairs
        for i in range(len(s) - 1):
            digit1, digit2 = s[i], s[i+1]
            
            # Check if the first digit is not equal to the second
            if digit1 != digit2:
                # Check if each digit appears exactly as many times as its numeric value
                if count[digit1] == int(digit1) and count[digit2] == int(digit2):
                    return digit1 + digit2
        
        return ""