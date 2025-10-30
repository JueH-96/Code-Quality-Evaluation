class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        # Number of times we can spend 5 liters from mainTank and inject 1 liter from additionalTank
        injections = min(mainTank // 5, additionalTank)
        
        # Distance covered by the full 5-liter cycles
        dist_injections = injections * 5 * 10
        
        # After these injections, each cycle effectively consumes 4 liters from mainTank
        # So we calculate how many liters remain in the mainTank
        leftover_main = mainTank - injections * 4
        
        # Distance covered by the leftover fuel in mainTank
        dist_leftover = leftover_main * 10
        
        return dist_injections + dist_leftover