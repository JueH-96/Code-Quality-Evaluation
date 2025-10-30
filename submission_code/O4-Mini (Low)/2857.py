class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        # Each time we burn 5 liters from main tank and if additionalTank >= 1,
        # we transfer 1 liter from additionalTank to mainTank.
        # Observe that for each such transfer ("chunk") net mainTank loss is 5 - 1 = 4 liters.
        # We can do at most mainTank//5 chunks, and also at most additionalTank chunks.
        chunks = min(mainTank // 5, additionalTank)
        # After doing 'chunks' transfers:
        #  - main tank lost 4 liters per chunk
        #  - additional tank lost 1 liter per chunk
        mainTank_after = mainTank - 4 * chunks
        # Total fuel burned = 5 liters per chunk + whatever remains in main tank
        total_liters_burned = 5 * chunks + mainTank_after
        # Mileage is 10 km per liter
        return total_liters_burned * 10