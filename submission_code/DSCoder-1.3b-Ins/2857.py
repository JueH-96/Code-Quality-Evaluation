class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        total_distance = 0
        while mainTank >= 5 and additionalTank >= 1:
            mainTank -= 5
            additionalTank -= 1
            total_distance += 10
        total_distance += mainTank + additionalTank * 1
        return total_distance