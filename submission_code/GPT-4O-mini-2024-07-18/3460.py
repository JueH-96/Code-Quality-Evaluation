class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10**9 + 7
        
        # Calculate factorials and inverse factorials
        fact = [1] * (n + 1)
        for i in range(2, n + 1):
            fact[i] = fact[i - 1] * i % MOD
        
        inv_fact = [1] * (n + 1)
        inv_fact[n] = pow(fact[n], MOD - 2, MOD)
        for i in range(n - 1, 0, -1):
            inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD
        
        # Count inversions for each prefix
        max_inversions = [0] * n
        for end, cnt in requirements:
            max_inversions[end] = cnt
        
        # Dynamic programming to count valid permutations
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        
        for length in range(1, n + 1):
            for inv in range(length + 1):
                dp[length][inv] = dp[length - 1][inv] * (length - 1) % MOD
                if inv > 0:
                    dp[length][inv] = (dp[length][inv] + dp[length - 1][inv - 1]) % MOD
        
        # Check if the requirements can be satisfied
        for end in range(n):
            if max_inversions[end] > end * (end + 1) // 2:
                return 0
            if max_inversions[end] < 0:
                return 0
        
        # Calculate the total number of valid permutations
        result = 1
        for end in range(n):
            if max_inversions[end] > 0:
                result = result * dp[end + 1][max_inversions[end]] % MOD
        
        return result