class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        # Precompute all possible powers of 5 (in decimal) up to the max binary number of length 15
        # Then store their binary representations in a set for quick lookup.
        powers_of_five_bin = set()
        val = 1
        while val <= 32767:
            powers_of_five_bin.add(bin(val)[2:])  # convert to binary string without '0b'
            val *= 5

        n = len(s)
        # dp[i] will hold the minimum beautiful partitions of s[i:], or -1 if impossible
        dp = [-1] * (n + 1)
        # Base case: no string left to partition => 0 partitions needed
        dp[n] = 0

        # Fill dp from right to left
        for i in range(n - 1, -1, -1):
            # If the current character is '0', skip (leading zero not allowed)
            if s[i] == '0':
                dp[i] = -1
                continue

            # Try all possible substrings s[i:j]
            dp[i] = -1
            for j in range(i + 1, n + 1):
                sub = s[i:j]
                # If sub is in our powers of five set and dp[j] is not -1, update dp[i]
                if sub in powers_of_five_bin and dp[j] != -1:
                    # If dp[i] is still -1, we set it directly; otherwise, take the minimum
                    if dp[i] == -1:
                        dp[i] = dp[j] + 1
                    else:
                        dp[i] = min(dp[i], dp[j] + 1)

        return dp[0]