class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        total_distance = 0
        fuel_consumed_since_last_transfer = 0
        
        while mainTank > 0:
            # Consume 1 liter of fuel
            mainTank -= 1
            total_distance += 10
            fuel_consumed_since_last_transfer += 1
            
            # Check if we need to transfer fuel
            if fuel_consumed_since_last_transfer == 5 and additionalTank > 0:
                mainTank += 1
                additionalTank -= 1
                fuel_consumed_since_last_transfer = 0
        
        return total_distance