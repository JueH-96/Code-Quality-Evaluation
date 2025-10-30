class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        total_distance = 0
        current_main = mainTank
        additional = additionalTank
        used_since_last = 0
        
        while current_main > 0:
            # Use 1 liter of fuel from the main tank
            current_main -= 1
            total_distance += 10
            used_since_last += 1
            
            # Check if we have used 5 liters since the last transfer
            if used_since_last == 5:
                # Transfer 1 liter from additional to main if possible
                if additional >= 1:
                    current_main += 1
                    additional -= 1
                # Reset the counter for used fuel since last transfer
                used_since_last = 0
        
        return total_distance