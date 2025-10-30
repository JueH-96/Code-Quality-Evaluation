class Solution:
    def getSmallestString(self, s: str) -> str:
        # Separate the digits into even and odd lists along with their indices
        even_digits = []
        odd_digits = []
        
        for i, char in enumerate(s):
            if int(char) % 2 == 0:
                even_digits.append((char, i))
            else:
                odd_digits.append((char, i))
        
        # Sort the even and odd digits to find the smallest possible arrangement
        even_digits.sort()
        odd_digits.sort()
        
        # Create a list from the original string to manipulate
        result = list(s)
        
        # Function to swap adjacent elements in the result based on sorted order
        def swap_adjacent(digits):
            for i in range(len(digits) - 1):
                if digits[i][1] + 1 == digits[i + 1][1]:  # Check if they are adjacent
                    # Perform the swap
                    result[digits[i][1]], result[digits[i + 1][1]] = result[digits[i + 1][1]], result[digits[i][1]]
                    return
        
        # Attempt to swap for even digits
        swap_adjacent(even_digits)
        # Attempt to swap for odd digits
        swap_adjacent(odd_digits)
        
        return ''.join(result)