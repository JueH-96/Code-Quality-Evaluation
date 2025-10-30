class Solution:
  def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
    """
    Calculates the maximum distance a truck can travel.

    The truck has a main tank and an additional tank.
    Mileage: 10 km per liter.
    Fuel transfer: For every 5 liters consumed from the main tank, 
                   if the additional tank has at least 1 liter, 
                   1 liter is transferred from the additional tank to the main tank.
    This transfer is immediate.
    """
    
    total_distance = 0  # Initialize total distance traveled
    
    # Loop as long as there is fuel in the main tank
    while mainTank > 0:
      if mainTank >= 5:
        # If mainTank has 5 or more liters, we consume 5 liters.
        # This consumption event might trigger a fuel transfer.
        
        # Consume 5 liters from the main tank
        mainTank -= 5
        total_distance += 5 * 10  # Travel 50 km for these 5 liters
        
        # Check if a transfer from the additional tank is possible and perform it
        if additionalTank >= 1:
          additionalTank -= 1  # Use 1 liter from additional tank
          mainTank += 1        # Add it to the main tank
      else:
        # If mainTank has less than 5 liters (but more than 0).
        # Consume all remaining fuel in the main tank.
        # No transfer will be triggered because 5 liters are not consumed in this step
        # to trigger the "every 5 liters consumed" condition.
        total_distance += mainTank * 10 # Travel mainTank * 10 km
        mainTank = 0  # Main tank becomes empty
        # The loop condition (mainTank > 0) will cause termination after this.
            
    return total_distance