class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        # Convert the string to a list for easier modifications
        chars = list(s)
        n = len(s)
        
        # We'll iterate over the first half of the string (up to the middle)
        for i in range(n // 2):
            j = n - 1 - i
            # If the characters at the mirrored positions differ,
            # choose the lexicographically smaller one for both.
            if chars[i] != chars[j]:
                smaller = min(chars[i], chars[j])
                chars[i] = smaller
                chars[j] = smaller
        
        # Join the list back into a string
        return "".join(chars)