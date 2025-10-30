class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        # Calculate the number of full 5-liter consumptions
        full_5L_consumptions = mainTank // 5
        # Calculate the actual transfers possible based on additionalTank availability
        actual_transfers = min(full_5L_consumptions, additionalTank)
        
        # Fuel consumed in full consumption events
        fuel_consumed = full_5L_consumptions * 5
        # Distance added from these consumptions
        distance = fuel_consumed * 10
        
        # Update mainTank after consumptions and transfers
        mainTank = mainTank - fuel_consumed + actual_transfers
        # Update additionalTank after transfers
        additionalTank = additionalTank - actual_transfers
        
        # Consume the remaining fuel in mainTank
        remaining_fuel = mainTank
        distance += remaining_fuel * 10
        
        return distance