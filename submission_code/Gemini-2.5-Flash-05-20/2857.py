class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        total_fuel_consumed = 0
        
        # Simulate the process of consuming fuel in blocks of 5 liters.
        # This loop continues as long as the main tank has at least 5 liters,
        # which is the threshold for checking fuel transfers.
        while mainTank >= 5:
            # Consume 5 liters from the main tank.
            # These 5 liters contribute to the total distance traveled.
            total_fuel_consumed += 5
            mainTank -= 5
            
            # After 5 liters are consumed, check if a transfer from the
            # additional tank is possible according to the rules.
            if additionalTank >= 1:
                # If there's fuel in the additional tank, transfer 1 liter
                # to the main tank.
                mainTank += 1
                additionalTank -= 1
        
        # After the loop, the main tank will have less than 5 liters (0 to 4 liters).
        # These remaining liters can still be used for travel, but they are not enough
        # to trigger another 5-liter consumption block and subsequent transfer.
        total_fuel_consumed += mainTank
        
        # The truck travels 10 km per liter of fuel.
        # Multiply the total fuel consumed by 10 to get the total distance.
        return total_fuel_consumed * 10