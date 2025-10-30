import collections

class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        # dp[i] will store the minimum number of balanced substrings for the prefix s[:i].
        # dp[0] is for an empty prefix, which requires 0 partitions.
        dp = [0] * (n + 1)

        # We iterate from i = 1 to n. dp[i] corresponds to the prefix s[:i].
        for i in range(1, n + 1):
            # Initialize dp[i] to a worst-case value. A partition of s[:i] into
            # i single-character substrings is always possible, and each is balanced.
            dp[i] = i

            # For each prefix s[:i], we check all possible last substrings s[j:i].
            # We iterate j backwards from i-1 to 0 to build the frequency map
            # of s[j:i] incrementally.
            counts = collections.Counter()
            for j in range(i - 1, -1, -1):
                counts[s[j]] += 1
                
                # A substring is balanced if all its characters have the same frequency.
                # This means the set of all frequency values in our counter
                # for the substring should have a size of 1.
                if len(set(counts.values())) == 1:
                    # If s[j:i] is balanced, we've found a potential partition.
                    # The number of substrings is 1 (for s[j:i]) plus the minimum
                    # number for the prefix s[:j], which is dp[j].
                    # Note: if j=0, this becomes 1 + dp[0] = 1, which correctly handles
                    # the case where the entire prefix s[:i] is one balanced substring.
                    dp[i] = min(dp[i], dp[j] + 1)
        
        # The final answer is the minimum partitions for the entire string s.
        return dp[n]