class Solution:
    def findValidPair(self, s: str) -> str:
        from collections import Counter
        
        # Count the occurrences of each digit
        count = Counter(s)
        
        # Iterate through the string to find the first valid pair
        for i in range(len(s) - 1):
            first_digit = s[i]
            second_digit = s[i + 1]
            
            # Check if the digits are different
            if first_digit != second_digit:
                # Check if the count of each digit matches its numeric value
                if count[first_digit] == int(first_digit) and count[second_digit] == int(second_digit):
                    return first_digit + second_digit
        
        # If no valid pair is found, return an empty string
        return ""