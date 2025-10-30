class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        distance = 0
        consumed_from_main = 0
        
        while mainTank > 0:
            # Consume 1 liter from main tank
            mainTank -= 1
            consumed_from_main += 1
            distance += 10
            
            # Check if we need to transfer fuel (every 5 liters consumed)
            if consumed_from_main == 5 and additionalTank > 0:
                # Transfer 1 liter from additional to main
                additionalTank -= 1
                mainTank += 1
                consumed_from_main = 0  # Reset counter
        
        return distance