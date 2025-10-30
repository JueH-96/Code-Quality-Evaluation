class Solution:
    def getSmallestString(self, s: str) -> str:
        # Convert the string to a list for easier manipulation
        s = list(s)
        n = len(s)
        
        # Function to check if a digit is even
        def is_even(digit):
            return int(digit) % 2 == 0
        
        # Iterate through the string
        for i in range(n - 1):
            # Check if the current digit and the next digit have the same parity
            if is_even(s[i]) == is_even(s[i + 1]):
                # Swap if the next digit is lexicographically smaller
                if s[i + 1] < s[i]:
                    s[i], s[i + 1] = s[i + 1], s[i]
        
        # Convert the list back to a string and return
        return ''.join(s)