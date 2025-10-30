class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        # Initialize the maximum odd binary number as the first character
        max_odd_bin = s[0]
        
        # Iterate over the string from the second character
        for i in range(1, len(s)):
            # If the current character is '1' and the current binary number is odd, update the maximum odd binary number
            if s[i] == '1' and int(max_odd_bin, 2) % 2 != 0:
                max_odd_bin = s[i]
        
        # Return the maximum odd binary number
        return max_odd_bin