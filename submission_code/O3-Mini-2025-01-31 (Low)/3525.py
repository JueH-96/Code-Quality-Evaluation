from typing import List
import math

class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        # We'll maintain three dp arrays:
        # dpA[i]: maximum energy sum we can get up to hour i if we drink drink A at hour i.
        # dpB[i]: maximum energy sum up to hour i if we drink drink B at hour i.
        # dpC[i]: maximum energy sum up to hour i if hour i is a cleansing hour (triggered by switching).
        #
        # Transitions:
        # From state A at hour i:
        #   (1) Continue same drink A at hour i+1: dpA[i+1] = dpA[i] + energyDrinkA[i+1]
        #   (2) Initiate a switch: then hour i+1 is forced cleansing, dpC[i+1] = max(dpC[i+1], dpA[i])
        # Similarly for state B.
        # From state C (cleansing) at hour i:
        #   We have just finished cleansing. At hour i+1 we can choose to drink either:
        #       dpA[i+1] = dpC[i] + energyDrinkA[i+1]
        #       dpB[i+1] = dpC[i] + energyDrinkB[i+1]
        #
        # Base condition: at hour 0, we can choose to take A or B.
        dpA = [-math.inf] * n
        dpB = [-math.inf] * n
        dpC = [-math.inf] * n
        
        dpA[0] = energyDrinkA[0]
        dpB[0] = energyDrinkB[0]
        # dpC[0] remains -inf because we don't start in cleansing.
        
        for i in range(n-1):
            # From state A at hour i:
            if dpA[i] != -math.inf:
                # Option 1: Continue using A next hour.
                candidate = dpA[i] + energyDrinkA[i+1]
                if candidate > dpA[i+1]:
                    dpA[i+1] = candidate
                # Option 2: Switch from A - so next hour (i+1) becomes cleansing.
                if dpA[i] > dpC[i+1]:
                    dpC[i+1] = dpA[i]
                    
            # From state B at hour i:
            if dpB[i] != -math.inf:
                # Option 1: Continue using B next hour.
                candidate = dpB[i] + energyDrinkB[i+1]
                if candidate > dpB[i+1]:
                    dpB[i+1] = candidate
                # Option 2: Switch from B - so next hour (i+1) is cleansing.
                if dpB[i] > dpC[i+1]:
                    dpC[i+1] = dpB[i]
            
            # From state C (cleansing) at hour i:
            if dpC[i] != -math.inf:
                # After cleansing, you can choose any drink in hour i+1.
                candidate = dpC[i] + energyDrinkA[i+1]
                if candidate > dpA[i+1]:
                    dpA[i+1] = candidate
                candidate = dpC[i] + energyDrinkB[i+1]
                if candidate > dpB[i+1]:
                    dpB[i+1] = candidate
        
        # The answer is the maximum among the possible states at hour n-1.
        return max(dpA[n-1], dpB[n-1], dpC[n-1])