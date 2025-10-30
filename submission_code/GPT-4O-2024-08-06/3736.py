class Solution:
    def findValidPair(self, s: str) -> str:
        from collections import Counter
        
        # Count the occurrences of each digit in the string
        digit_count = Counter(s)
        
        # Traverse the string to find the first valid pair
        for i in range(len(s) - 1):
            first_digit = s[i]
            second_digit = s[i + 1]
            
            # Check if the digits are different
            if first_digit != second_digit:
                # Check if each digit appears in the string as many times as its numeric value
                if digit_count[first_digit] == int(first_digit) and digit_count[second_digit] == int(second_digit):
                    return first_digit + second_digit
        
        # If no valid pair is found, return an empty string
        return ""