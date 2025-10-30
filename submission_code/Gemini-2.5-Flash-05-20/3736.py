import collections

class Solution:
    def findValidPair(self, s: str) -> str:
        # Step 1: Pre-calculate the total occurrences of each digit in the string s.
        # This is efficient using collections.Counter.
        # Example: for s = "2523533", counts will be {'2': 2, '5': 2, '3': 3}
        counts = collections.Counter(s)

        # Step 2: Iterate through the string to find adjacent pairs.
        # We iterate up to len(s) - 2 because we need s[i] and s[i+1].
        for i in range(len(s) - 1):
            char1 = s[i]
            char2 = s[i+1]

            # Condition 1: The first digit must not be equal to the second.
            if char1 == char2:
                continue  # If they are equal, this pair is not valid, move to the next.

            # Convert the character digits to their integer numeric values.
            # The problem constraints state s only consists of digits from '1' to '9'.
            num1 = int(char1)
            num2 = int(char2)

            # Condition 2: Each digit in the pair must appear in s exactly as many times as its numeric value.
            # Check this for both digits in the current pair.
            is_char1_valid_by_count = (counts[char1] == num1)
            is_char2_valid_by_count = (counts[char2] == num2)

            # If both digits satisfy their count criteria, we have found a valid pair.
            # Since we are traversing from left to right, this is the *first* valid pair.
            if is_char1_valid_by_count and is_char2_valid_by_count:
                return char1 + char2  # Return the pair as a string.
        
        # If the loop completes without finding any valid pair, return an empty string.
        return ""