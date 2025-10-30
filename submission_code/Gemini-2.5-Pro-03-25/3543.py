class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        """
        Counts the number of substrings of s that satisfy the k-constraint.

        A binary string satisfies the k-constraint if either the number of 0's
        is at most k, or the number of 1's is at most k.

        Args:
            s: The input binary string.
            k: The constraint integer.

        Returns:
            The number of substrings satisfying the k-constraint.
        """
        n = len(s)
        satisfying_substring_count = 0 # Initialize count of valid substrings

        # Iterate through all possible starting positions (i) of a substring
        for i in range(n):
            count0 = 0 # Count of '0's in the substring starting at index i
            count1 = 0 # Count of '1's in the substring starting at index i

            # Iterate through all possible ending positions (j) of a substring starting at i
            # This inner loop considers substrings s[i], s[i..i+1], ..., s[i..n-1]
            for j in range(i, n):
                # Update the counts for the current character s[j] which extends
                # the substring from s[i..j-1] to s[i..j]
                if s[j] == '0':
                    count0 += 1
                else: # s[j] must be '1' since it's a binary string
                    count1 += 1

                # Check if the current substring s[i..j] meets the k-constraint
                # The constraint is satisfied if the count of '0's is at most k,
                # OR the count of '1's is at most k.
                if count0 <= k or count1 <= k:
                    satisfying_substring_count += 1 # Increment the counter if constraint is met
                # If neither count is <= k, the substring does not satisfy the constraint,
                # and we simply continue to the next substring (by incrementing j).

        # Return the total count of substrings that satisfy the k-constraint
        return satisfying_substring_count