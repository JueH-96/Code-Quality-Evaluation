class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        if n < 4:
            return 0
        if n == 4:
            return 12
        # Calculate the number of ways to place 'l', 'e', 'e', 't' in n positions
        ways_to_place = (comb(n, 4) * 2 * 24) % MOD
        # Calculate the number of ways to fill the remaining positions
        ways_to_fill = pow(26, n - 4, MOD)
        # Calculate the total number of good strings
        total_good_strings = (ways_to_place * ways_to_fill) % MOD
        return total_good_strings