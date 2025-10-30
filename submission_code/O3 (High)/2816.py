class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        # Work with a mutable list of characters
        chars = list(s)
        n = len(chars)

        # Process symmetric pairs
        for i in range(n // 2):
            j = n - 1 - i                  # symmetric index
            # To keep the number of changes minimal, we may change at most
            # one of the two characters in a mismatching pair.
            # Picking the smaller of the two gives the lexicographically
            # smallest overall result.
            smaller = min(chars[i], chars[j])
            chars[i] = chars[j] = smaller

        # Middle character (if any) never needs to change for minimal edits.
        return ''.join(chars)