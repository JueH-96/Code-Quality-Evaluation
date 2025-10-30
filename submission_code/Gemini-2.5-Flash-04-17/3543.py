class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        """
        Counts the number of substrings of a binary string s that satisfy the k-constraint.
        A substring satisfies the k-constraint if the number of 0's in it is at most k
        OR the number of 1's in it is at most k.

        Args:
            s: The input binary string.
            k: The integer constraint.

        Returns:
            The number of substrings satisfying the k-constraint.
        """
        n = len(s)
        count = 0
        # Iterate through all possible starting indices i of a substring
        for i in range(n):
            # Initialize counts of zeros and ones for the current substring starting at i
            zeros = 0
            ones = 0
            # Iterate through all possible ending indices j of a substring starting at i
            # The substring is s[i:j+1]
            for j in range(i, n):
                # Add the character s[j] to the current substring and update counts
                if s[j] == '0':
                    zeros += 1
                else: # s[j] == '1'
                    ones += 1
                
                # Check the k-constraint:
                # A substring satisfies the k-constraint if the number of 0's <= k
                # OR the number of 1's <= k.
                if zeros <= k or ones <= k:
                    # If the constraint is satisfied, increment the valid substring counter
                    count += 1
                else:
                    # If the constraint is NOT satisfied (i.e., zeros > k AND ones > k),
                    # then this substring s[i:j+1] is invalid.
                    # Consider any longer substring starting at the same index i,
                    # say s[i:j'+1] where j' > j. This longer substring contains s[i:j+1].
                    # The number of zeros in s[i:j'+1] will be >= zeros, and the number
                    # of ones will be >= ones. Since zeros > k and ones > k, it follows
                    # that zeros' >= zeros > k and ones' >= ones > k.
                    # Therefore, any longer substring starting at i will also NOT satisfy
                    # the k-constraint.
                    # We can optimize by breaking the inner loop (over j) and moving to
                    # the next starting index i, as no further substrings starting at i
                    # will be valid.
                    break # Optimization

        # Return the total count of substrings that satisfy the k-constraint
        return count