class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        distance = 0
        # While there's enough fuel in the main tank for a complete 5 liter usage cycle
        while mainTank >= 5:
            # Consume 5 liters to cover 50 km (5 * 10 km per liter)
            mainTank -= 5
            distance += 50
            # For each consumption of 5 liters, if additional tank has fuel, inject 1 liter to the main tank.
            if additionalTank > 0:
                additionalTank -= 1
                mainTank += 1
        # Use any remaining fuel in the main tank
        distance += mainTank * 10
        return distance

# Example usage:
# sol = Solution()
# print(sol.distanceTraveled(5, 10))  # Expected output: 60
# print(sol.distanceTraveled(1, 2))   # Expected output: 10