class Solution:
    def findValidPair(self, s: str) -> str:
        # Count frequency of each digit
        freq = {}
        for char in s:
            freq[char] = freq.get(char, 0) + 1
        
        # Check adjacent pairs from left to right
        for i in range(len(s) - 1):
            first_digit = s[i]
            second_digit = s[i + 1]
            
            # Check if digits are different
            if first_digit != second_digit:
                # Check if each digit appears exactly as many times as its value
                first_value = int(first_digit)
                second_value = int(second_digit)
                
                if (freq.get(first_digit, 0) == first_value and 
                    freq.get(second_digit, 0) == second_value):
                    return first_digit + second_digit
        
        return ""