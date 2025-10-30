class Solution:
	def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
		k = min((mainTank - 1) // 4, additionalTank)
		return 10 * (mainTank + k)