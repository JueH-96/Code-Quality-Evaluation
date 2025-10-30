class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        remaining_main = mainTank
        remaining_add = additionalTank
        distance = 0
        total_used = 0
        
        while remaining_main > 0:
            remaining_main -= 1
            distance += 10
            total_used += 1
            
            if total_used % 5 == 0:
                if remaining_add >= 1:
                    remaining_add -= 1
                    remaining_main += 1
        
        return distance