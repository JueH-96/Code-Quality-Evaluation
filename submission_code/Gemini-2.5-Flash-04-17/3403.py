import math

class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        # dp[i] is the minimum number of balanced substrings
        # that partition the prefix s[:i]
        # dp[0] is the base case for an empty string, requiring 0 partitions
        dp = [math.inf] * (n + 1)
        dp[0] = 0

        # Helper function to check if a list of counts represents a balanced string
        # A string is balanced if all characters that appear in it
        # occur the same number of times.
        def is_balanced(counts):
            first_count = -1
            for count in counts:
                if count > 0:
                    if first_count == -1:
                        # This is the count of the first character encountered
                        first_count = count
                    elif count != first_count:
                        # Found another character with a different non-zero count
                        return False
            # If loop finishes, all non-zero counts were the same (or no characters present)
            return True


        # Iterate through all possible end points i for the prefix s[:i]
        # i represents the length of the prefix s[:i], or the index *after* the prefix ends.
        # So i ranges from 1 to n (inclusive).
        for i in range(1, n + 1):
            # counts will store character frequencies for the substring s[j:i]
            # This counter needs to be fresh for each potential ending index i,
            # but accumulates characters as we consider different starting indices j.
            counts = [0] * 26

            # Iterate through all possible start points j for the last substring s[j:i]
            # The substring is s[j]...s[i-1].
            # We iterate j backwards from i-1 down to 0.
            # When j=i-1, substring is s[i-1:i] (length 1)
            # When j=i-2, substring is s[i-2:i] (length 2)
            # ...
            # When j=0, substring is s[0:i] (length i)
            for j in range(i - 1, -1, -1):
                # Add the character s[j] (the leftmost character of the current substring s[j:i])
                # to the frequency counts.
                char_index = ord(s[j]) - ord('a')
                counts[char_index] += 1

                # Check if the current substring s[j:i] (from index j up to i-1) is balanced.
                if is_balanced(counts):
                    # If s[j:i] is balanced, it means we can form a valid partition
                    # where the last substring is s[j:i].
                    # The number of partitions for s[:i] would be the minimum partitions
                    # required for the prefix s[:j] (which is dp[j]) plus the one new partition s[j:i].
                    # We want the minimum number of partitions, so we take the minimum
                    # over all valid start points j.
                    dp[i] = min(dp[i], dp[j] + 1)

        # The minimum number of balanced substrings for the entire string s[:n] is dp[n].
        return dp[n]