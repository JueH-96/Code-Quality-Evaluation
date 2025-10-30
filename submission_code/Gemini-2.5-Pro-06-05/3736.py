import collections

class Solution:
    def findValidPair(self, s: str) -> str:
        """
        Finds the first valid pair of adjacent digits in a string.

        A pair of adjacent digits d1d2 is valid if:
        1. d1 is not equal to d2.
        2. The count of d1 in the string s is equal to the numeric value of d1.
        3. The count of d2 in the string s is equal to the numeric value of d2.

        Args:
          s: The input string consisting of digits '1' through '9'.

        Returns:
          The first valid pair as a two-character string, or an empty string
          if no valid pair is found.
        """
        
        # Step 1: Count the frequency of each digit in the entire string.
        # collections.Counter is an efficient and Pythonic way to do this.
        counts = collections.Counter(s)
        
        # Step 2: Iterate through the string to find the first valid adjacent pair.
        # We loop up to the second-to-last character to form pairs.
        for i in range(len(s) - 1):
            d1 = s[i]
            d2 = s[i+1]
            
            # Condition 1: The digits must be different.
            if d1 == d2:
                continue
                
            # Convert character digits to their integer values for comparison.
            val1 = int(d1)
            val2 = int(d2)
            
            # Condition 2 & 3: The count of each digit in the pair must match its numeric value.
            # Note: counts[d1] is safe because d1 is guaranteed to be a key in the Counter
            # as it's a character from the original string 's'.
            if counts[d1] == val1 and counts[d2] == val2:
                # We've found the first valid pair. Return it immediately.
                return d1 + d2
                
        # If the loop finishes without finding any valid pair, return an empty string.
        return ""