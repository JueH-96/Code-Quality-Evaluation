class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        
        # dpA, dpB, and dpS represent dp[i][0], dp[i][1], and dp[i][2] respectively
        # dpA = max total energy ending with drinking A at hour i
        # dpB = max total energy ending with drinking B at hour i
        # dpS = max total energy ending with skipping hour i
        
        dpA = energyDrinkA[0]
        dpB = energyDrinkB[0]
        dpS = 0
        
        for i in range(1, n):
            nextA = max(dpA, dpS) + energyDrinkA[i]  # continue A or come from skip
            nextB = max(dpB, dpS) + energyDrinkB[i]  # continue B or come from skip
            nextS = max(dpA, dpB, dpS)               # skip this hour
            
            dpA, dpB, dpS = nextA, nextB, nextS
        
        return max(dpA, dpB, dpS)