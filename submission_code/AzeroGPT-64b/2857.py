class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        
        # Calculate the number of times fuel will be transferred from additional tank to the main tank.
        num_transfers = min(mainTank // 5, additionalTank)
        
        # Deduct the 5 litres spent and add the 1 litre transfer for each transfer event.
        mainTank -= 5 * num_transfers
        mainTank += num_transfers
        
        # Calculate the total fuel used, which is initial main tank fuel minus the 5 litres spent (but not replaced) plus the unused additional tank.
        total_fuel_used = mainTank + additionalTank - num_transfers
        
        # Considering the max distance that can be covered is 10 km/litre, return the total distance covered.
        return total_fuel_used * 10