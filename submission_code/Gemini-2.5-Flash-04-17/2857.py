class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        """
        Calculates the maximum distance a truck can travel by simulating
        fuel consumption and transfer step by step.

        Args:
            mainTank: Fuel in the main tank in liters (1 <= mainTank <= 100).
            additionalTank: Fuel in the additional tank in liters (1 <= additionalTank <= 100).

        Returns:
            The maximum distance traveled in kilometers.
        """
        total_distance = 0

        # The truck can travel as long as there is fuel in the main tank.
        # The process continues until the main tank is empty.
        while mainTank > 0:
            # Check if there are at least 5 liters in the main tank.
            # If yes, we can consume a 5-liter block, which triggers a potential transfer check.
            if mainTank >= 5:
                # Consume 5 liters from the main tank.
                mainTank -= 5
                # Add the distance traveled by consuming 5 liters (5 liters * 10 km/liter).
                total_distance += 50

                # After using 5 liters, check if a transfer from the additional tank occurs.
                # A transfer happens if the additional tank has at least 1 liter.
                if additionalTank >= 1:
                    # Transfer 1 liter from the additional tank to the main tank.
                    mainTank += 1
                    additionalTank -= 1
            else:
                # If the main tank has less than 5 liters (but more than 0),
                # consume the remaining fuel.
                # This consumption does not complete a 5-liter block, so no
                # new transfer is triggered by this specific consumption event.
                total_distance += mainTank * 10
                # The main tank becomes empty, which will terminate the loop
                # in the next iteration check.
                mainTank = 0

        return total_distance