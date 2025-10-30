import collections

class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7

        # dp[k] will store the length of the string generated from a single 'z'
        # after k transformations. The maximum k we need is t.
        dp = [0] * (t + 1)
        
        # Base Case: dp[0]
        # This corresponds to a character that becomes 'z' and has 0 transformations left.
        # The length is 1 (for the character "z"). This is also needed for the recurrence.
        dp[0] = 1

        # Base Cases: dp[1] to dp[25]
        # For 1 <= k <= 25, 'z' becomes "ab", then "bc", ..., up to "yz".
        # The length of the resulting string is always 2.
        for k in range(1, min(t + 1, 26)):
            dp[k] = 2
        
        # Recurrence Relation for k >= 26: dp[k] = dp[k-26] + dp[k-25]
        for k in range(26, t + 1):
            dp[k] = (dp[k - 26] + dp[k - 25]) % MOD

        total_length = 0
        ord_z = ord('z')
        
        # Using collections.Counter is an optimization for long strings.
        char_counts = collections.Counter(s)

        for char, count in char_counts.items():
            dist_to_z = ord_z - ord(char)

            if t <= dist_to_z:
                # The character does not become 'z' within t transformations.
                # Its length contribution remains 1.
                length_contrib = 1
            else:
                # The character becomes 'z' after dist_to_z steps.
                # We find the length after the remaining (t - dist_to_z) transformations
                # from our precomputed dp table.
                remaining_t = t - dist_to_z
                length_contrib = dp[remaining_t]
            
            # Add the total contribution of this character type to the total length.
            total_length = (total_length + count * length_contrib) % MOD
            
        return total_length