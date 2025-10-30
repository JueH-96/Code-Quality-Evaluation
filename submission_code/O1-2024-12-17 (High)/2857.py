class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        # Each time 5 liters are used from the main tank, we can transfer 1 liter
        # from the additional tank (if available). The number of such transfers
        # is limited by how many times we can subtract 5 liters from mainTank,
        # as well as how much fuel is in the additional tank.
        
        # Calculate how many transfers can occur
        transfers = min(mainTank // 5, additionalTank)
        
        # Total liters used is the original mainTank plus any successful transfers
        total_liters_used = mainTank + transfers
        
        # Each liter provides 10 km of travel
        return total_liters_used * 10