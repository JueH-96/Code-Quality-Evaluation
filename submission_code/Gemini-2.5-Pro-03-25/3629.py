import sys
# Setting a reasonable recursion depth, although the solution is iterative.
# sys.setrecursionlimit(200000) 

class Solution:
    """
    Solves the problem of finding the length of a string after t transformations.
    The transformation rule is:
    - 'z' is replaced by "ab".
    - Any other character 'c' is replaced by the next character in the alphabet (e.g., 'a' -> 'b', 'b' -> 'c').
    These transformations are applied simultaneously to all characters in the string.
    The process is repeated t times.
    The goal is to find the length of the resulting string modulo 10^9 + 7.

    The approach uses dynamic programming. Let L(c, k) be the length of the string generated 
    by starting with a single character c and applying the transformation k times.
    The total length of the final string starting from s is the sum of L(c, t) for all characters c in s.

    The recurrence relations for L(c, k) are:
    - If c != 'z', L(c, k) = L(next(c), k-1) for k > 0.
    - If c == 'z', L('z', k) = L('a', k-1) + L('b', k-1) for k > 0.
    - Base case: L(c, 0) = 1 for any character c.

    Let dp[k][i] represent L(c, k) where c is the i-th character of the alphabet (0-indexed, 'a'=0, ..., 'z'=25).
    The recurrence relations translate to:
    - dp[k][i] = dp[k-1][i+1] for i < 25, k > 0.
    - dp[k][25] = (dp[k-1][0] + dp[k-1][1]) % MOD for k > 0.
    - dp[0][i] = 1 for all i in [0, 25].

    We are interested in dp[t][i] for the characters i present in the initial string s.
    Through analysis, we find that the length contribution of a character depends on when it turns into 'z'.
    Character i takes k_i = 25 - i steps to become 'z'.
    
    Let F_k = dp[k][25] be the length generated starting from 'z' after k steps.
    The sequence F_k follows a linear recurrence:
    - F_0 = 1 (Length of 'z' at time 0)
    - F_1 = L('z', 1) = L('a', 0) + L('b', 0) = 1 + 1 = 2
    - ...
    - F_k = 2 for 1 <= k <= 25
    - F_26 = L('z', 26) = L('a', 25) + L('b', 25) = L('z', 0) + L('z', 1) = 1 + F_1 = 1 + 2 = 3
    - F_k = (F_{k-26} + F_{k-25}) % MOD for k >= 27. This is derived from dp[k][25] = dp[k-1][0] + dp[k-1][1] and relating dp[k-1][0] and dp[k-1][1] back to F values.
      Specifically, dp[k-1][0] = dp[k-1-25][25] = F_{k-26} for k-1 >= 25 (i.e., k >= 26).
      And dp[k-1][1] = dp[k-1-24][25] = F_{k-25} for k-1 >= 24 (i.e., k >= 25).
      The combined recurrence holds for k >= 27.

    For any character c (with 0-based index i), its length contribution L(c, t) after t steps is:
    - L(c, t) = 1 if t < k_i = 25 - i. (The character hasn't become 'z' yet).
    - L(c, t) = F_{t - k_i} if t >= k_i = 25 - i. (The character became 'z' at step k_i, and then evolved for t - k_i steps).

    The total length is the sum of L(c, t) for all characters c in the initial string s, calculated modulo MOD.
    We can precompute the F_k sequence up to t and then sum the contributions.
    """
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7

        # Handle the base case t=0: no transformations applied, length is unchanged.
        if t == 0:
            return len(s)

        # Precompute the F_k sequence up to t using dynamic programming.
        # F_vals[k] will store the value of F_k. The array size needs to be t+1.
        max_k = t
        F_vals = [0] * (max_k + 1) # Indices from 0 to t.

        # Initialize base cases for the F_k sequence derived from analysis.
        # F_0 = 1
        if max_k >= 0:
             F_vals[0] = 1
        
        # F_k = 2 for 1 <= k <= 25. Use min check for small t values to avoid index out of bounds.
        for k in range(1, min(26, max_k + 1)):
            F_vals[k] = 2
        
        # F_26 = 3. Check if t >= 26 to ensure index 26 is valid.
        if max_k >= 26:
            F_vals[26] = 3
        
        # Compute F_k for k >= 27 using the linear recurrence relation F_k = F_{k-26} + F_{k-25}.
        # Check if t >= 27 to ensure the loop runs and accesses valid indices.
        if max_k >= 27:
            for k in range(27, max_k + 1):
                # The indices k-26 and k-25 are guaranteed to be non-negative since k >= 27.
                # They are also <= k-2 < k, so they fall within the already computed range.
                F_vals[k] = (F_vals[k - 26] + F_vals[k - 25]) % MOD

        # Calculate the total length of the string after t transformations.
        total_length = 0
        for char in s:
            # Get the 0-based index i for the character c.
            i = ord(char) - ord('a')
            
            # Calculate k_i, the number of steps required for character i to become 'z'.
            k_i = 25 - i 
            
            # Determine the length contribution of this character after t steps.
            if t < k_i:
                # If t is less than the steps required to reach 'z', the character simply shifts
                # along the alphabet, its length contribution remains 1.
                length_contribution = 1
            else:
                # If t is sufficient or more steps than required to reach 'z',
                # the character became 'z' at step k_i.
                # Its length contribution after t total steps is determined by the length 
                # generated from 'z' over the remaining t - k_i steps. This is F_{t - k_i}.
                # The index t - k_i is guaranteed to be non-negative since t >= k_i.
                length_contribution = F_vals[t - k_i]
            
            # Add the character's contribution to the total length, applying modulo arithmetic.
            total_length = (total_length + length_contribution) % MOD

        return total_length