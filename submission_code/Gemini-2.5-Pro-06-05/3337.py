class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        """
        Calculates the total number of substrings of s that start and end with c.

        The problem can be solved by realizing it's a combinatorial question.
        If the character 'c' appears 'k' times in the string 's', any pair of these
        occurrences can define a substring that starts and ends with 'c'.

        Let the count of character 'c' be k.
        - The first 'c' can form k substrings (with itself and all subsequent 'c's).
        - The second 'c' can form k-1 substrings.
        - ...
        - The k-th 'c' can form 1 substring (with itself).

        The total is the sum 1 + 2 + ... + k, which is the k-th triangular number.
        The formula for this sum is k * (k + 1) / 2.

        Args:
            s: The input string.
            c: The character to check for at the start and end of substrings.

        Returns:
            The total number of such substrings.
        """
        
        # Step 1: Count the occurrences of the character 'c'.
        # The s.count() method is efficient for this, running in O(N) time.
        count_c = s.count(c)
        
        # If the character does not appear, no such substrings can be formed.
        if count_c == 0:
            return 0
            
        # Step 2: Apply the formula for the sum of the first 'count_c' integers.
        # We use integer division // to ensure the result is an integer.
        # Python's integers handle arbitrary size, so overflow is not a concern
        # for the intermediate product, even with s.length up to 10^5.
        # For count_c = 10^5, the result is ~5 * 10^9, which fits in a 64-bit integer.
        return count_c * (count_c + 1) // 2