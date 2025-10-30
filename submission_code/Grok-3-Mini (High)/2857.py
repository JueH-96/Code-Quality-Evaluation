class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        transfers = min(mainTank // 5, additionalTank)
        total_consumed = mainTank + transfers
        return total_consumed * 10