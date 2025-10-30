class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        
        # dp[i][j][k][l] represents the number of good strings of length i with j 'l's, k 'e's, and l 't's
        dp = [[[[-1 for _ in range(2)] for _ in range(3)] for _ in range(2)] for _ in range(n + 1)]
        
        def count_strings(length, l_count, e_count, t_count):
            if length == 0:
                return 1 if l_count >= 1 and e_count >= 2 and t_count >= 1 else 0
            
            if dp[length][l_count][e_count][t_count] != -1:
                return dp[length][l_count][e_count][t_count]
            
            total = 0
            # Add 'l'
            if l_count < 1:
                total += count_strings(length - 1, l_count + 1, e_count, t_count)
            # Add 'e'
            if e_count < 2:
                total += count_strings(length - 1, l_count, e_count + 1, t_count)
            # Add 't'
            if t_count < 1:
                total += count_strings(length - 1, l_count, e_count, t_count + 1)
            # Add any other character
            total += 23 * count_strings(length - 1, l_count, e_count, t_count)
            
            dp[length][l_count][e_count][t_count] = total % MOD
            return dp[length][l_count][e_count][t_count]
        
        return count_strings(n, 0, 0, 0)