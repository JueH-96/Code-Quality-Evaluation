class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        # dp[i] => minimum partitions for s[:i] (i from 0 to n)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0

        # A helper function to check if the substring s[i:j] is balanced.
        def is_balanced(i, j):
            # We can count frequency for substring s[i:j].
            freq = {}
            for k in range(i, j):
                freq[s[k]] = freq.get(s[k], 0) + 1
            # All characters that appear must have the same frequency.
            freqs = list(freq.values())
            if not freqs:
                return True  # should not happen as substring is non-empty
            first = freqs[0]
            for count in freqs:
                if count != first:
                    return False
            return True

        # DP to compute the minimum number of balanced substrings partitioning s[0:i].
        for i in range(n):
            # if dp[i] is reachable, try to add a balanced substring from i to j.
            if dp[i] < float('inf'):
                freq = {}
                # Instead of recalculating from scratch each time, we'll maintain frequency while expanding the substring.
                # And we'll check if balanced condition holds at each extension.
                for j in range(i, n):
                    char = s[j]
                    freq[char] = freq.get(char, 0) + 1
                    # The substring s[i:j+1] is balanced if all freq values in freq dictionary are equal.
                    # We can check by verifying that min == max among the values.
                    values = list(freq.values())
                    if min(values) == max(values):
                        dp[j+1] = min(dp[j+1], dp[i] + 1)
                        
        return dp[n]
      
# Testing the solution with the provided examples
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    s1 = "fabccddg"
    print(sol.minimumSubstringsInPartition(s1))  # Expected output: 3
    
    # Example 2:
    s2 = "abababaccddb"
    print(sol.minimumSubstringsInPartition(s2))  # Expected output: 2