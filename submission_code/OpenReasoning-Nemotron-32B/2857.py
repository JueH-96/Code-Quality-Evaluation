class Solution:
	def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
		transfers = min(additionalTank, (mainTank - 1) // 4)
		total_fuel = mainTank + transfers
		return total_fuel * 10