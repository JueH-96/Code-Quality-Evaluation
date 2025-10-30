class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        dp = {}

        def solve(idx, last_drink, current_energy):
            if idx == n:
                return current_energy
            
            if (idx, last_drink) in dp:
                return dp[(idx, last_drink)]

            max_energy = 0
            
            #Option 1: Continue with the same drink
            if last_drink == 0:
                max_energy = max(max_energy, solve(idx + 1, 0, current_energy + energyDrinkA[idx]))
            elif last_drink == 1:
                max_energy = max(max_energy, solve(idx + 1, 1, current_energy + energyDrinkB[idx]))

            #Option 2: Switch drinks (lose one hour)
            if last_drink == 0:
                max_energy = max(max_energy, solve(idx + 1, 1, current_energy + energyDrinkB[idx]))
            elif last_drink == 1:
                max_energy = max(max_energy, solve(idx + 1, 0, current_energy + energyDrinkA[idx]))

            
            dp[(idx, last_drink)] = max_energy
            return max_energy

        #Start with either drink A or B
        result = max(solve(0, 0, 0), solve(0,1,0))
        return result