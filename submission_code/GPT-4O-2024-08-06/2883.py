class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        # Precompute the binary representations of powers of 5 up to a reasonable limit
        powers_of_5 = set()
        i = 0
        while True:
            power_of_5 = 5 ** i
            binary_representation = bin(power_of_5)[2:]  # Get the binary representation without the '0b' prefix
            if len(binary_representation) > 15:
                break
            powers_of_5.add(binary_representation)
            i += 1

        # Dynamic programming approach to find the minimum number of partitions
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Base case: no characters, no partitions

        for i in range(1, n + 1):
            for j in range(i):
                substring = s[j:i]
                if substring in powers_of_5:
                    dp[i] = min(dp[i], dp[j] + 1)

        return dp[n] if dp[n] != float('inf') else -1