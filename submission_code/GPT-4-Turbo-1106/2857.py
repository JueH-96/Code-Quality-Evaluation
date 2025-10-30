class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        distance = 0
        while mainTank > 0:
            # If mainTank has at least 5 liters, consume 5 liters and possibly get 1 liter from additionalTank
            if mainTank >= 5:
                mainTank -= 4  # 5 liters consumed, 1 liter possibly added
                distance += 50  # 5 liters give 50 km
                if additionalTank > 0:
                    additionalTank -= 1  # Transfer 1 liter from additionalTank to mainTank
            else:
                # If mainTank has less than 5 liters, consume them all
                distance += mainTank * 10
                mainTank = 0
        return distance