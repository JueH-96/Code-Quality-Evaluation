class Solution:
	def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
		n = len(energyDrinkA)
		prev2_0 = energyDrinkA[0]
		prev2_1 = energyDrinkB[0]
		prev1_0 = prev2_0 + energyDrinkA[1]
		prev1_1 = prev2_1 + energyDrinkB[1]
		
		for i in range(2, n):
			curr0 = max(prev1_0 + energyDrinkA[i], prev2_1 + energyDrinkA[i])
			curr1 = max(prev1_1 + energyDrinkB[i], prev2_0 + energyDrinkB[i])
			prev2_0, prev2_1 = prev1_0, prev1_1
			prev1_0, prev1_1 = curr0, curr1
		
		return max(prev1_0, prev1_1)