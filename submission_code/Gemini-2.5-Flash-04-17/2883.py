class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        n = len(s)

        # If the string starts with '0', it's impossible to partition
        # A beautiful string cannot have leading zeros (unless it's "0", but "0" isn't a power of 5)
        # The binary representation of any positive integer starts with '1'.
        if s[0] == '0':
            return -1

        # Precompute beautiful strings: binary representations of powers of 5
        # We only need powers of 5 whose binary string length is <= 15 (the max length of s)
        # 5^0 = 1 (bin "1", len 1)
        # 5^1 = 5 (bin "101", len 3)
        # 5^2 = 25 (bin "11001", len 5)
        # 5^3 = 125 (bin "1111101", len 7)
        # 5^4 = 625 (bin "1001110001", len 10)
        # 5^5 = 3125 (bin "110000110101", len 12)
        # 5^6 = 15625 (bin "11110100001001", len 14)
        # 5^7 = 78125 (bin "100110001010001", len 17) -> too long
        beautiful_strings = {
            "1",
            "101",
            "11001",
            "1111101",
            "1001110001",
            "110000110101",
            "11110100001001"
        }

        # dp[i] is the minimum number of beautiful substrings to partition s[0:i]
        # Initialize with infinity (a large number representing impossibility)
        # dp size needs to be n + 1 to store result for prefix of length n (s[0:n])
        dp = [float('inf')] * (n + 1)

        # Base case: An empty prefix (s[0:0]) requires 0 beautiful substrings
        dp[0] = 0

        # Iterate through the string, considering prefixes of increasing length
        # i represents the length of the prefix s[0:i], or the index *after* the prefix ends
        for i in range(1, n + 1):
            # For each prefix s[0:i], consider all possible split points j (0 <= j < i)
            # A split at j means s[j:i] is the last substring in the partition
            # The prefix s[0:j] must then be partitionable into dp[j] beautiful substrings
            for j in range(i):
                # sub is the potential last beautiful substring s[j:i]
                sub = s[j:i]

                # Check if sub is a beautiful string (binary representation of a power of 5)
                # The check `sub in beautiful_strings` automatically covers the "no leading zeros" rule
                # for strings of length > 1, and for "1" for length 1.
                # We don't need an explicit check for sub[0] == '0' here because none of the strings
                # in `beautiful_strings` start with '0' (except implicitly length-1 "1").
                if sub in beautiful_strings:
                    # If s[0:j] could be partitioned (dp[j] is not infinity)
                    if dp[j] != float('inf'):
                        # We can partition s[0:i] using partitions of s[0:j] plus the current substring s[j:i]
                        # Update dp[i] with the minimum number of partitions found so far
                        dp[i] = min(dp[i], dp[j] + 1)

        # The final result is the minimum partitions for the whole string s[0:n]
        # If dp[n] is still infinity, it means the string cannot be partitioned
        return dp[n] if dp[n] != float('inf') else -1