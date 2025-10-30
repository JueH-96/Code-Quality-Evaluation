class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        INF = float('-inf')
        # DP variables: dp[drink][k]
        # drink: 0 for A, 1 for B
        # k: 0 for not cleansing, 1 for cleansing to switch to this drink
        dp = [[0, INF], [0, INF]]  # dp[drink][k]
        dp[0][0] = energyDrinkA[0]
        dp[1][0] = energyDrinkB[0]
        for i in range(1, n):
            new_dp = [[INF, INF], [INF, INF]]
            for drink in [0,1]:
                # Continue with same drink
                new_dp[drink][0] = max(new_dp[drink][0], dp[drink][0] + (energyDrinkA[i] if drink ==0 else energyDrinkB[i]))
                # Start cleansing to switch to other drink
                other_drink = 1 - drink
                new_dp[other_drink][1] = max(new_dp[other_drink][1], dp[drink][0])
                # If we were in cleansing state to this drink, we can start drinking it now
                new_dp[drink][0] = max(new_dp[drink][0], dp[drink][1] + (energyDrinkA[i] if drink ==0 else energyDrinkB[i]))
            dp = new_dp
        return max(dp[0][0], dp[1][0])