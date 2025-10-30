class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10**9 + 7
        
        # Calculate the number of ways to assign performers to stages
        # Each performer can be assigned to any of the x stages
        stage_assignments = x ** n
        
        # Calculate the number of ways to assign scores to each band
        # Each band (stage) can receive any score from 1 to y
        # We need to consider only the stages that have at least one performer
        # The number of non-empty stages can be from 1 to min(n, x)
        score_assignments = 0
        for non_empty_stages in range(1, min(n, x) + 1):
            # Calculate the number of ways to choose non_empty_stages from x stages
            # and assign performers to these stages
            # This is equivalent to distributing n performers into non_empty_stages
            # which is given by the Stirling numbers of the second kind
            # multiplied by the number of ways to choose non_empty_stages from x
            # and multiplied by the number of ways to assign scores to these stages
            score_assignments += (y ** non_empty_stages) * self.stirling_second_kind(n, non_empty_stages) * self.comb(x, non_empty_stages)
            score_assignments %= MOD
        
        return (stage_assignments * score_assignments) % MOD
    
    def stirling_second_kind(self, n, k):
        # Stirling numbers of the second kind S(n, k) is the number of ways to partition
        # a set of n objects into k non-empty subsets
        if k == 0 or k > n:
            return 0
        if k == n:
            return 1
        if k == 1:
            return 1
        # Use dynamic programming to calculate S(n, k)
        S = [[0] * (k + 1) for _ in range(n + 1)]
        S[0][0] = 1
        for i in range(1, n + 1):
            for j in range(1, k + 1):
                S[i][j] = j * S[i - 1][j] + S[i - 1][j - 1]
                S[i][j] %= 10**9 + 7
        return S[n][k]
    
    def comb(self, n, k):
        # Calculate n choose k
        if k > n:
            return 0
        if k == 0 or k == n:
            return 1
        # Use a simple iterative approach to calculate n choose k
        result = 1
        for i in range(1, k + 1):
            result = result * (n - i + 1) // i
        return result