class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        # Convert string to list for easy modification
        chars = list(s)
        n = len(chars)
        
        # Only need to iterate through the first half
        for i in range(n // 2):
            j = n - 1 - i
            if chars[i] == chars[j]:
                # Already a matching pair, nothing to do
                continue
            # Choose the lexicographically smaller character
            # and set both positions to it (cost = 1 operation)
            smaller = min(chars[i], chars[j])
            chars[i] = smaller
            chars[j] = smaller
        
        # Join back into a string
        return "".join(chars)