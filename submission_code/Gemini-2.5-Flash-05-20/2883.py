class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        n = len(s)

        # Precompute beautiful binary strings.
        # A string is beautiful if:
        # 1. It doesn't contain leading zeros. (Handled by checking substring[0] == '0' below)
        # 2. It's the binary representation of a number that is a power of 5.
        
        # Max length of s is 15. Powers of 5 that result in binary strings <= 15 bits:
        # 5^0 = 1    -> "1" (length 1)
        # 5^1 = 5    -> "101" (length 3)
        # 5^2 = 25   -> "11001" (length 5)
        # 5^3 = 125  -> "1111101" (length 7)
        # 5^4 = 625  -> "1001110001" (length 10)
        # 5^5 = 3125 -> "110000111101" (length 12)
        # 5^6 = 15625 -> "11110011010001" (length 14)
        # 5^7 = 78125 -> "1001100010000101" (length 17 - too long)
        
        beautiful_strings = set()
        for k in range(7): # Covers 5^0 to 5^6
            power_of_5 = 5**k
            binary_repr = bin(power_of_5)[2:] # Convert to binary string, remove "0b" prefix
            beautiful_strings.add(binary_repr)

        # dp[i] stores the minimum number of beautiful substrings to partition s[0...i-1]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0 # Base case: empty prefix needs 0 substrings

        # Iterate through all possible end points `i` for a substring (exclusive)
        for i in range(1, n + 1):
            # Iterate through all possible start points `j` for the current substring `s[j:i]`
            for j in range(i):
                current_substring = s[j:i]
                
                # Condition 1: Substring must not have leading zeros.
                # If a substring starts with '0', it's invalid unless it's just "0" itself.
                # However, "0" is not a power of 5, so no beautiful string will be "0".
                # Therefore, any substring starting with '0' is not beautiful.
                if current_substring[0] == '0':
                    continue
                
                # Check if the substring is one of the precomputed beautiful binary representations
                if current_substring in beautiful_strings:
                    # If the prefix s[0...j-1] was partitionable (dp[j] is not infinity)
                    if dp[j] != float('inf'):
                        # Update dp[i] with the minimum number of partitions
                        # (partitions for s[0...j-1] + 1 for the current substring s[j:i])
                        dp[i] = min(dp[i], dp[j] + 1)
        
        # If dp[n] is still infinity, it means no valid partition exists for the entire string s.
        return dp[n] if dp[n] != float('inf') else -1