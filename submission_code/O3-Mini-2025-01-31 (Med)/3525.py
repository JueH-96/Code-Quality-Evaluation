from typing import List

class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        # We use dpA[i] to denote the maximum energy boost we can achieve starting from hour i
        # if the planned drink at hour i is A.
        # Similarly, dpB[i] for drink B.
        # When we are in a “switch” decision, the rule forces us to lose a whole hour.
        #
        # Thus at hour i for a given drink d (0 for A, 1 for B):
        #   Option 1 (continue): drink d in hour i and get its energy,
        #       then remain on d in hour i+1.
        #           value = energy[d][i] + dp_d[i+1]
        #   Option 2 (switch): cancel drinking at hour i (cleansing hour with 0 energy),
        #       and at hour i+1 drink the other type.
        #           value = energy[other][i+1] + dp_other[i+2]   (only allowed if i+1 < n)
        #
        # Base cases:
        # dpA[n] = dpB[n] = 0. To safeguard i+2 indexing, we fill arrays with size n+2.
        
        dpA = [0] * (n+2)
        dpB = [0] * (n+2)
        
        # Process from the last hour backwards
        for i in range(n-1, -1, -1):
            # When scheduled to drink A at hour i:
            cont_A = energyDrinkA[i] + dpA[i+1]  # option 1: continue drinking A
            if i+1 < n:  # option 2: switch from A to B (i is cleansing hour, drink B at i+1)
                switch_A = energyDrinkB[i+1] + dpB[i+2]
            else:
                switch_A = -10**18  # not allowed if no subsequent hour
            dpA[i] = cont_A if cont_A >= switch_A else switch_A
            
            # Similarly, when scheduled to drink B at hour i:
            cont_B = energyDrinkB[i] + dpB[i+1]  # option 1: continue drinking B
            if i+1 < n:  # option 2: switch from B to A
                switch_B = energyDrinkA[i+1] + dpA[i+2]
            else:
                switch_B = -10**18
            dpB[i] = cont_B if cont_B >= switch_B else switch_B
        
        # We can start with either drink at hour 0.
        return dpA[0] if dpA[0] >= dpB[0] else dpB[0]


# The following code is for internal testing.
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    energyDrinkA = [1, 3, 1]
    energyDrinkB = [3, 1, 1]
    # Expected output: 5 (drinking the same drink always gives 5)
    print(sol.maxEnergyBoost(energyDrinkA, energyDrinkB))
    
    # Example 2:
    energyDrinkA = [4, 1, 1]
    energyDrinkB = [1, 1, 3]
    # Expected output: 7 (drink A at hour 0, switch to B for hour 2)
    print(sol.maxEnergyBoost(energyDrinkA, energyDrinkB))