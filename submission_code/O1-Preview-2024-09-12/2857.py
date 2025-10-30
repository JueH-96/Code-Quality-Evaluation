class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        number_of_transfers = min(mainTank // 5, additionalTank)
        total_fuel = mainTank + number_of_transfers
        return total_fuel * 10