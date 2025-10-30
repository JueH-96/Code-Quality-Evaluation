class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        total_consumed = 0
        
        # Continue while we can make transfers (need 5L in main tank and 1L in additional)
        while mainTank >= 5 and additionalTank >= 1:
            # Consume 5 liters from main tank
            mainTank -= 5
            total_consumed += 5
            
            # Transfer 1 liter from additional to main tank
            additionalTank -= 1
            mainTank += 1
        
        # Consume any remaining fuel in main tank
        total_consumed += mainTank
        
        # Convert to distance (10 km per liter)
        return total_consumed * 10