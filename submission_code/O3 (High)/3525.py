from typing import List

class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        """
        dpA – best total boost up to the current hour if we drink A now
        dpB – best total boost up to the current hour if we drink B now
        dpN – best total boost up to the current hour if we skip this hour
        A switch is modelled by the transition  drink → skip → other drink
        """
        n = len(energyDrinkA)
        
        NEG = float('-inf')       # helper for impossible states
        
        dpA = energyDrinkA[0]     # drink A at hour 0
        dpB = energyDrinkB[0]     # drink B at hour 0
        dpN = NEG                 # we cannot start with a skip
        
        for i in range(1, n):
            # drink A at hour i:
            #   either we were already drinking A,
            #   or we skipped the previous hour after drinking B
            newA = max(dpA, dpN) + energyDrinkA[i]
            
            # drink B at hour i (symmetric)
            newB = max(dpB, dpN) + energyDrinkB[i]
            
            # skip hour i:
            #   only needed when we want to switch drinks
            newN = max(dpA, dpB)
            
            dpA, dpB, dpN = newA, newB, newN
        
        return int(max(dpA, dpB, dpN))