class Solution:
    def getSmallestString(self, s: str) -> str:
        # Python strings are immutable, so we work with a list of characters.
        chars = list(s)
        n = len(s)

        # Iterate through adjacent pairs to find the first beneficial swap.
        for i in range(n - 1):
            # Check if both digits have the same parity (both even or both odd).
            if int(chars[i]) % 2 == int(chars[i+1]) % 2:
                # Check if swapping would make the string lexicographically smaller.
                if chars[i] > chars[i+1]:
                    # Perform the swap.
                    chars[i], chars[i+1] = chars[i+1], chars[i]
                    
                    # We are allowed at most one swap. The first beneficial swap
                    # from the left yields the lexicographically smallest result.
                    # So, we can stop searching.
                    break
        
        # Join the characters back into a string and return.
        return "".join(chars)