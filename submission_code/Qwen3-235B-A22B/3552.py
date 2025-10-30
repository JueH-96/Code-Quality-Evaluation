class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        if k == 1:
            return '9' * n
        
        m = (n + 1) // 2
        pow_10 = [1] * n
        for i in range(1, n):
            pow_10[i] = (pow_10[i-1] * 10) % k
        
        coeff = [0] * m
        if n % 2 == 0:
            for i in range(m):
                rev_pos = n - 1 - i
                coeff[i] = (pow_10[i] + pow_10[rev_pos]) % k
        else:
            for i in range(m):
                if i < m - 1:
                    rev_pos = n - 1 - i
                    coeff[i] = (pow_10[i] + pow_10[rev_pos]) % k
                else:
                    coeff[i] = pow_10[i] % k
        
        # Initialize DP table
        dp = [[False] * k for _ in range(m + 2)]
        dp[m][0] = True  # Base case
        
        for i in range(m - 1, -1, -1):
            current_coeff = coeff[i]
            mod_vals = [(d * current_coeff) % k for d in range(10)]
            for s in range(k):
                if dp[i + 1][s]:
                    for mod_val in mod_vals:
                        r = (mod_val + s) % k
                        dp[i][r] = True
        
        # Greedily build the prefix
        prefix = []
        current_residue = 0
        for i in range(m):
            for d in range(9, -1, -1):
                if i == 0 and d == 0:
                    continue  # Skip leading zero
                contribution = (d * coeff[i]) % k
                new_residue = (current_residue + contribution) % k
                required_remainder = (-new_residue) % k
                if dp[i + 1][required_remainder]:
                    prefix.append(str(d))
                    current_residue = new_residue
                    break
        
        # Construct the full palindrome
        prefix_str = ''.join(prefix)
        if n % 2 == 0:
            full_pal = prefix_str + prefix_str[::-1]
        else:
            full_pal = prefix_str + prefix_str[:-1][::-1]
        return full_pal