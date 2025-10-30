class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        distance = 0
        while mainTank > 0:
            if mainTank >= 5:
                # Travel for 5 liters worth of distance
                distance += 5 * 10
                mainTank -= 5
                # Check if we can transfer fuel from additional tank
                if additionalTank > 0:
                    mainTank += 1
                    additionalTank -= 1
            else:
                # Travel for the remaining fuel in the main tank
                distance += mainTank * 10
                mainTank = 0
        return distance