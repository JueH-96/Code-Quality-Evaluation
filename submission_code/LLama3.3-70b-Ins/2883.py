class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        # Generate all powers of 5 in binary up to 15 bits
        powers_of_five = set()
        for i in range(15):
            num = 5 ** i
            binary = bin(num)[2:]
            powers_of_five.add(binary)

        # Initialize a list to store the minimum number of substrings for each prefix of s
        dp = [float('inf')] * (len(s) + 1)
        dp[0] = 0

        # Iterate over the string
        for i in range(1, len(s) + 1):
            # Check all substrings ending at the current position
            for j in range(i):
                # Get the current substring
                substring = s[j:i]
                # Check if the substring is beautiful
                if substring[0] == '1' and substring in powers_of_five:
                    # Update the minimum number of substrings
                    dp[i] = min(dp[i], dp[j] + 1)

        # Return the minimum number of substrings or -1 if it's impossible to partition
        return dp[-1] if dp[-1] != float('inf') else -1