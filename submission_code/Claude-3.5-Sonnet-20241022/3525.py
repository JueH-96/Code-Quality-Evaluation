class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        
        # dp[i][j] represents max energy at index i when last drink was j
        # j=0 for drink A, j=1 for drink B
        dp = [[0] * 2 for _ in range(n)]
        
        # Initialize first hour - can drink either A or B
        dp[0][0] = energyDrinkA[0]  # Drink A
        dp[0][1] = energyDrinkB[0]  # Drink B
        
        # For each hour
        for i in range(1, n):
            # If we drink A in current hour
            # We can either continue A from previous hour
            # Or switch from B with 1 hour gap
            dp[i][0] = max(dp[i-1][0] + energyDrinkA[i],
                          (dp[i-2][1] + energyDrinkA[i] if i > 1 else energyDrinkA[i]))
            
            # If we drink B in current hour  
            # We can either continue B from previous hour
            # Or switch from A with 1 hour gap
            dp[i][1] = max(dp[i-1][1] + energyDrinkB[i],
                          (dp[i-2][0] + energyDrinkB[i] if i > 1 else energyDrinkB[i]))
        
        # Return max of last hour drinking either A or B
        return max(dp[n-1][0], dp[n-1][1])