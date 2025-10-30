class Solution:
    def findValidPair(self, s: str) -> str:
        for i in range(len(s) - 1):
            first_digit = s[i]
            second_digit = s[i+1]
            
            if first_digit != second_digit:
                first_digit_count = s.count(first_digit)
                second_digit_count = s.count(second_digit)
                
                if first_digit_count == int(first_digit) and second_digit_count == int(second_digit):
                    return first_digit + second_digit
        
        return ""