class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        current_main = mainTank
        current_additional = additionalTank
        distance = 0
        
        while current_main > 0:
            use = min(5, current_main)
            distance += use * 10
            current_main -= use
            
            if use == 5 and current_additional >= 1:
                current_additional -= 1
                current_main += 1
        
        return distance