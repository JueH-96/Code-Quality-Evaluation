class Solution:
    def findValidPair(self, s: str) -> str:
        # Count frequency of each digit
        count = {}
        for char in s:
            count[char] = count.get(char, 0) + 1
        
        # Traverse the string and check each adjacent pair
        for i in range(len(s) - 1):
            first = s[i]
            second = s[i + 1]
            
            # Check if the digits are different
            if first != second:
                # Check if each digit appears exactly as many times as its numeric value
                if count[first] == int(first) and count[second] == int(second):
                    return first + second
        
        return ""