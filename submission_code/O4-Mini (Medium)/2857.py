class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        # Each time we consume 5L from the main tank, we can transfer 1L from the additional tank,
        # but only if there is still fuel left in the additional tank.
        # The number of possible transfers is limited by how many full 5L chunks
        # we can consume from the main tank, which is (mainTank-1)//4,
        # and by the amount of fuel in the additional tank.
        transfers = min(additionalTank, (mainTank - 1) // 4)
        total_liters_used = mainTank + transfers
        return total_liters_used * 10