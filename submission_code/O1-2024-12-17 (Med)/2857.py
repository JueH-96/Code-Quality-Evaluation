class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        # Determine how many times we can use 5 liters from the main tank 
        # and get a bonus liter from the additional tank:
        x = min(mainTank // 5, additionalTank)
        
        # For each of those x times, effectively one extra liter goes to the main tank.
        # The total liters used by the main tank is mainTank + x,
        # and each liter provides 10 km.
        return 10 * (mainTank + x)