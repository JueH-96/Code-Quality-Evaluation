class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10**9 + 7
        
        # Calculate the number of ways to assign performers to stages
        stage_assignments = pow(x, n, MOD)
        
        # Calculate the number of ways to assign scores to bands
        band_scores = pow(y, x, MOD)
        
        # Multiply the number of stage assignments and band scores to get the total number of ways
        return (stage_assignments * band_scores) % MOD