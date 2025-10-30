class Solution:
    def findValidPair(self, s: str) -> str:
        from collections import Counter
        
        # Count the occurrences of each digit in the string
        count = Counter(s)
        
        # Traverse the string to find the first valid pair
        for i in range(len(s) - 1):
            first_digit = s[i]
            second_digit = s[i + 1]
            
            # Check if the digits are different and their counts match their values
            if first_digit != second_digit:
                if count[first_digit] == int(first_digit) and count[second_digit] == int(second_digit):
                    return first_digit + second_digit
        
        return ""