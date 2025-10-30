class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        distance = 0
        while mainTank > 0:
            if mainTank >= 5:
                distance += 50  # 5 liters used for 50 km
                mainTank -= 5
                if additionalTank > 0:
                    mainTank += 1  # transfer 1 liter from additional tank
                    additionalTank -= 1
            else:
                distance += mainTank * 10  # remaining fuel for distance
                mainTank = 0  # main tank is now empty
        return distance