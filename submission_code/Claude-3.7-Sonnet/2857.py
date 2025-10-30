class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        total_distance = 0
        fuel_used_from_main = 0
        
        while mainTank > 0:
            # Use 1 liter of fuel
            mainTank -= 1
            fuel_used_from_main += 1
            total_distance += 10  # 10 km per liter
            
            # Check if we've used a multiple of 5 liters and can inject
            if fuel_used_from_main % 5 == 0 and additionalTank > 0:
                additionalTank -= 1
                mainTank += 1
        
        return total_distance