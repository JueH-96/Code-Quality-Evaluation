from collections import Counter

class Solution:
    def findValidPair(self, s: str) -> str:
        # Calculate the frequency of each digit in the string
        counts = Counter(s)

        # Iterate through the string to check adjacent pairs
        # The loop runs up to len(s) - 2 to get adjacent pairs s[i] and s[i+1]
        for i in range(len(s) - 1):
            digit1 = s[i]
            digit2 = s[i+1]

            # Condition 1: The two adjacent digits must not be equal
            if digit1 != digit2:
                # Convert digits to their integer values
                num1 = int(digit1)
                num2 = int(digit2)

                # Check conditions 2 & 3: Each digit appears in s exactly as many times as its numeric value
                # We use counts.get(digit, 0) to safely get the count.
                # Although given constraints (digits 1-9, length >= 2), digits from s will always be in counts.
                count1 = counts.get(digit1, 0)
                count2 = counts.get(digit2, 0)

                if count1 == num1 and count2 == num2:
                    # If all conditions are met, we've found the first valid pair.
                    # Return the pair as a string.
                    return digit1 + digit2

        # If the loop completes without finding a valid pair, return an empty string.
        return ""