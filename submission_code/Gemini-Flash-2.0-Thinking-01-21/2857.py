class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        distance = 0
        fuel_consumed = 0
        while mainTank > 0:
            mainTank -= 1
            distance += 10
            fuel_consumed += 1
            if fuel_consumed == 5:
                if additionalTank >= 1:
                    mainTank += 1
                    additionalTank -= 1
                fuel_consumed = 0
        return distance