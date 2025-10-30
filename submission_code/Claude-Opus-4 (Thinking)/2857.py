class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        totalFuelUsed = 0
        
        while mainTank > 0:
            if mainTank >= 5:
                # Use 5 liters
                mainTank -= 5
                totalFuelUsed += 5
                
                # Transfer 1 liter from additional tank if available
                if additionalTank > 0:
                    mainTank += 1
                    additionalTank -= 1
            else:
                # Use remaining fuel
                totalFuelUsed += mainTank
                mainTank = 0
        
        return totalFuelUsed * 10