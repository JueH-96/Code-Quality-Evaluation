class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        total_distance = 0
        while mainTank > 0:
            if mainTank >= 5:
                mainTank -= 5
                if additionalTank >= 1:
                    mainTank += 1
                    additionalTank -= 1
            total_distance += mainTank * 10
            mainTank = 0
        return total_distance