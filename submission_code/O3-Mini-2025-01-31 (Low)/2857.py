class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        distance = 0
        # While we can use at least 5 liters from the main tank.
        while mainTank >= 5:
            # Use 5 liters and travel 50 km
            mainTank -= 5
            distance += 50
            # If additionalTank has fuel, transfer 1 liter.
            if additionalTank >= 1:
                additionalTank -= 1
                mainTank += 1
        # After the loop, use remaining fuel in mainTank.
        distance += mainTank * 10
        return distance