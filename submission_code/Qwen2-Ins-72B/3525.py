class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        dp = [[0]*3 for _ in range(n+1)]
        
        dp[0][0] = 0 # no drink
        dp[0][1] = energyDrinkA[0] # drink A
        dp[0][2] = energyDrinkB[0] # drink B
        
        for i in range(1, n):
            dp[i][0] = max(dp[i-1][0], dp[i-1][1], dp[i-1][2]) # no drink
            dp[i][1] = max(dp[i-1][0], dp[i-1][2]) + energyDrinkA[i] # drink A
            dp[i][2] = max(dp[i-1][0], dp[i-1][1]) + energyDrinkB[i] # drink B
            
        return max(dp[n-1][0], dp[n-1][1], dp[n-1][2])