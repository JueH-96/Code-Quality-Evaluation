class Solution:
    def findValidPair(self, s: str) -> str:
        # Count frequency of each digit in the string
        freq = {}
        for digit in s:
            freq[digit] = freq.get(digit, 0) + 1
        
        # Check each adjacent pair from left to right
        for i in range(len(s) - 1):
            first_digit = s[i]
            second_digit = s[i + 1]
            
            # Check if digits are different
            if first_digit != second_digit:
                # Check if each digit appears exactly as many times as its value
                if (freq[first_digit] == int(first_digit) and 
                    freq[second_digit] == int(second_digit)):
                    return first_digit + second_digit
        
        # No valid pair found
        return ""