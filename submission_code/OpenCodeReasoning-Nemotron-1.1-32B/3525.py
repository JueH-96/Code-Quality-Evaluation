class Solution:
	def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
		n = len(energyDrinkA)
		if n == 0:
			return 0
		a0 = energyDrinkA[0]
		b0 = energyDrinkB[0]
		ans = max(a0, b0)
		if n == 1:
			return ans
		a1 = a0 + energyDrinkA[1]
		b1 = b0 + energyDrinkB[1]
		ans = max(ans, a1, b1)
		if n == 2:
			return ans
		
		a_prev2, a_prev1 = a0, a1
		b_prev2, b_prev1 = b0, b1
		
		for i in range(2, n):
			a_curr = max(a_prev1 + energyDrinkA[i], b_prev2 + energyDrinkA[i])
			b_curr = max(b_prev1 + energyDrinkB[i], a_prev2 + energyDrinkB[i])
			ans = max(ans, a_curr, b_curr)
			a_prev2, a_prev1 = a_prev1, a_curr
			b_prev2, b_prev1 = b_prev1, b_curr
		
		return ans