class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        used = 0
        distance = 0
        while mainTank > 0:
            mainTank -= 1
            distance += 10
            used += 1
            if used == 5:
                if additionalTank >= 1:
                    mainTank += 1
                    additionalTank -= 1
                used = 0
        return distance