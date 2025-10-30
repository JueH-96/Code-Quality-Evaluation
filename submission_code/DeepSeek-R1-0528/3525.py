class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        a_pp = energyDrinkA[0]
        b_pp = energyDrinkB[0]
        a_p = a_pp + energyDrinkA[1]
        b_p = b_pp + energyDrinkB[1]
        
        for i in range(2, n):
            new_a = energyDrinkA[i] + max(a_p, b_pp)
            new_b = energyDrinkB[i] + max(b_p, a_pp)
            a_pp, b_pp = a_p, b_p
            a_p, b_p = new_a, new_b
        
        return max(a_p, b_p)