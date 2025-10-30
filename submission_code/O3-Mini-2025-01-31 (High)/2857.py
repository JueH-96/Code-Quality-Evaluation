class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        distance = 0
        
        # While there is enough fuel to use 5 liters from the main tank,
        # simulate a consumption cycle where the truck travels 50 km.
        while mainTank >= 5:
            mainTank -= 5
            distance += 50  # Travels 5 liters * 10 km per liter
            # If there is fuel in the additional tank, transfer 1 liter.
            if additionalTank > 0:
                mainTank += 1
                additionalTank -= 1
        
        # After all possible cycles, add the distance from the remaining fuel.
        distance += mainTank * 10
        return distance