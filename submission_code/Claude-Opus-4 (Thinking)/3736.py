class Solution:
    def findValidPair(self, s: str) -> str:
        # Count occurrences of each digit
        count = {}
        for digit in s:
            count[digit] = count.get(digit, 0) + 1
        
        # Check each adjacent pair
        for i in range(len(s) - 1):
            first = s[i]
            second = s[i + 1]
            
            # Check if digits are different
            if first != second:
                # Check if each digit appears exactly as many times as its value
                if count[first] == int(first) and count[second] == int(second):
                    return first + second
        
        return ""