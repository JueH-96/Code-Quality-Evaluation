import math # This import is not actually needed for the final solution.

class Solution:
    """
    Calculates the maximum distance a truck can travel given initial fuel
    in the main and additional tanks.

    The truck has a mileage of 10 km per liter.
    Whenever 5 liters of fuel get used up in the main tank, if the additional
    tank has at least 1 liter of fuel, 1 liter of fuel will be transferred
    from the additional tank to the main tank.
    """
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        """
        Simulates the fuel consumption and transfer process to find the total distance.

        Args:
            mainTank: The initial fuel in the main tank (liters).
            additionalTank: The initial fuel in the additional tank (liters).

        Returns:
            The maximum distance the truck can travel in kilometers.
        """
        # Initialize total distance traveled
        total_distance = 0

        # Simulate the process step-by-step based on consumption of 5-liter chunks.
        # The loop continues as long as the main tank has enough fuel (>= 5 liters)
        # to potentially trigger a transfer after consumption.
        while mainTank >= 5:
            # Consume 5 liters from the main tank
            mainTank -= 5
            # Add 50 km to the total distance (5 liters * 10 km/liter)
            total_distance += 50

            # Check if there is fuel in the additional tank to perform a transfer
            if additionalTank > 0:
                # Transfer 1 liter from the additional tank to the main tank
                additionalTank -= 1
                # The transferred fuel is added to the main tank
                mainTank += 1

        # After the loop, the main tank contains less than 5 liters (0 to 4).
        # The truck consumes all the remaining fuel in the main tank.
        # Calculate the distance covered by this remaining fuel.
        total_distance += mainTank * 10 # remaining liters * 10 km/liter

        # Return the total calculated distance
        return total_distance