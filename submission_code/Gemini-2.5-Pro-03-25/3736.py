import collections

class Solution:
    """
    Finds the first valid adjacent pair of digits in a string based on specific criteria.

    A valid pair (d1, d2) at index i and i+1 satisfies:
    1. d1 != d2
    2. The count of digit d1 in the entire string s equals the numeric value of d1.
    3. The count of digit d2 in the entire string s equals the numeric value of d2.
    """
    def findValidPair(self, s: str) -> str:
        """
        Finds the first valid pair in the string s.

        Args:
            s: The input string consisting of digits '1' to '9'.

        Returns:
            The first valid pair as a two-character string, or an empty string ""
            if no valid pair exists.
        """
        
        # Step 1: Calculate the frequency of each digit in the entire string s.
        # collections.Counter is efficient for this.
        counts = collections.Counter(s)
        
        # Step 2: Iterate through all adjacent pairs of digits in the string.
        # The loop goes up to len(s) - 2 because we need pairs (s[i], s[i+1]).
        # The last possible starting index for a pair is len(s) - 2.
        for i in range(len(s) - 1):
            d1 = s[i]
            d2 = s[i+1]
            
            # Step 3: Check the first condition for a valid pair:
            # The two adjacent digits must not be equal.
            if d1 == d2:
                continue  # Skip if digits are the same

            # Step 4: Check the second condition for a valid pair:
            # Each digit's count in the string must match its numeric value.
            
            # Convert digit characters to their integer values.
            val1 = int(d1)
            val2 = int(d2)
            
            # Get the counts of these digits from our pre-calculated frequency map.
            # Counter returns 0 if a key is not found, but based on constraints 
            # and the logic, if a digit is part of a potential pair, it must exist
            # in the counts map.
            count1 = counts[d1]
            count2 = counts[d2]
            
            # Check if both digits satisfy the frequency condition.
            if count1 == val1 and count2 == val2:
                # Found the first valid pair, return it as a string.
                # s[i:i+2] creates the two-character substring for the pair.
                return s[i:i+2] 
                
        # Step 5: If the loop completes without finding a valid pair, return an empty string.
        return ""