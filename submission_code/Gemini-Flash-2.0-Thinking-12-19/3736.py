from collections import Counter

class Solution:
    def findValidPair(self, s: str) -> str:
        # Calculate the frequency of each digit in the string
        # Using Counter is efficient for this step.
        counts = Counter(s)

        # Iterate through the string to find adjacent pairs.
        # We iterate up to len(s) - 2 because we need s[i] and s[i+1].
        for i in range(len(s) - 1):
            digit1 = s[i]
            digit2 = s[i+1]

            # Condition 1: The first digit is not equal to the second.
            if digit1 == digit2:
                continue # If digits are the same, it's not a valid pair.

            # Condition 2: Each digit appears in s exactly as many times as its numeric value.
            # Convert digit characters to integers to get their numeric values.
            num1 = int(digit1)
            num2 = int(digit2)

            # Get the counts of the digits from the frequency map.
            # We use .get() just in case, although with the constraints and problem logic,
            # digits in a pair must appear, so their counts would be >= 1.
            count1 = counts.get(digit1, 0)
            count2 = counts.get(digit2, 0)

            # Check if the counts match the numeric values for both digits.
            if count1 == num1 and count2 == num2:
                # Found the first valid pair according to the rules.
                # Return the pair as a string.
                return digit1 + digit2

        # If the loop finishes without finding a valid pair, return an empty string.
        return ""