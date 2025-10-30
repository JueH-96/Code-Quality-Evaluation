class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        k = min(additionalTank, (mainTank - 1) // 4)
        total_fuel = mainTank + k
        return total_fuel * 10