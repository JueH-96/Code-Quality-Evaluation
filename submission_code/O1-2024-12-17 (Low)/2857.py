class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        distance = 0
        
        # Use fuel in chunks of 5 liters from mainTank as long as possible
        while mainTank >= 5:
            # Travel distance for these 5 liters
            distance += 5 * 10  # 5 liters * 10 km/liter
            mainTank -= 5
            
            # Inject 1 liter from additionalTank if available
            if additionalTank > 0:
                mainTank += 1
                additionalTank -= 1
        
        # Use the remaining liters in the mainTank
        distance += mainTank * 10
        
        return distance