class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        total_distance = 0
        mileage = 10  # km per liter

        while mainTank > 0:
            # Calculate distance traveled with the current mainTank fuel
            distance = min(mainTank, 5) * mileage
            total_distance += distance

            # Calculate fuel used
            fuel_used = distance / mileage
            mainTank -= fuel_used

            # Transfer fuel from additionalTank if possible
            if fuel_used == 5 and additionalTank > 0:
                mainTank += 1
                additionalTank -= 1

        return total_distance