class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        total_distance = 0
        fuel_used = 0
        
        while mainTank > 0:
            # Use 1 liter from main tank
            mainTank -= 1
            fuel_used += 1
            total_distance += 10
            
            # Check if we've used 5 liters and can transfer from additional tank
            if fuel_used == 5 and additionalTank > 0:
                mainTank += 1
                additionalTank -= 1
                fuel_used = 0
                
        return total_distance