import collections

class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        
        # dp[i] will store the minimum number of balanced substrings
        # required to partition the prefix s[0...i-1].
        # dp[0] = 0, as an empty string requires 0 substrings.
        dp = [float('inf')] * (n + 1)
        dp[0] = 0

        # Iterate through all possible end points (i) of a partition.
        # i represents the length of the prefix s[:i].
        for i in range(1, n + 1):
            # For each i, iterate through all possible start points (j)
            # of the last substring s[j...i-1].
            # This means we are considering s[j...i-1] as a candidate for the last balanced substring.
            
            counts = collections.defaultdict(int)
            
            # Iterate j from i-1 down to 0, to efficiently update counts for s[j...i-1].
            # This means substring s[j...i-1] starts at j and ends at i-1.
            for j in range(i - 1, -1, -1):
                char = s[j]
                counts[char] += 1

                # Check if the current substring s[j...i-1] is balanced.
                # A string is balanced if all characters in it occur the same number of times.
                
                is_current_substring_balanced = True
                first_freq = -1 # This will store the frequency of the first character encountered in 'counts'.

                # Iterate through the frequencies of characters in the current substring.
                # All frequencies must be equal to be balanced.
                for freq in counts.values():
                    if first_freq == -1:
                        # Set the reference frequency with the first character encountered.
                        first_freq = freq
                    elif freq != first_freq:
                        # If any character has a different frequency, it's not balanced.
                        is_current_substring_balanced = False
                        break
                
                # If s[j...i-1] is balanced, then we can potentially form a partition ending at i-1.
                # The minimum substrings for s[0...i-1] would be
                # dp[j] (substrings for s[0...j-1]) + 1 (for s[j...i-1]).
                if is_current_substring_balanced:
                    if dp[j] != float('inf'): # Ensure dp[j] is reachable (i.e., prefix s[0...j-1] can be partitioned)
                        dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n]