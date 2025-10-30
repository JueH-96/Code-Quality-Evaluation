class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        count = 0

        # Iterate through all possible starting positions of a substring.
        for i in range(n):
            zeros = 0
            ones = 0
            # For each starting position 'i', extend the substring to the right with index 'j'.
            for j in range(i, n):
                # Update the counts of '0's and '1's for the current substring s[i..j].
                if s[j] == '0':
                    zeros += 1
                else:
                    ones += 1
                
                # Check if the k-constraint is satisfied for the current substring.
                # A substring is valid if the number of 0s is at most k,
                # OR the number of 1s is at most k.
                if zeros <= k or ones <= k:
                    count += 1
                else:
                    # Optimization: If the substring becomes invalid (both counts > k),
                    # any further extension from this starting point 'i' will also be invalid,
                    # as the counts of zeros and ones will only increase.
                    # Thus, we can break the inner loop and move to the next 'i'.
                    break
            
        return count