class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        dp = [0] * n

        # Initialize the dp array with the energy of the first magician
        dp[0] = energy[0]

        # Fill the dp array
        for i in range(1, n):
            dp[i] = energy[i] + max(dp[max(0, i - k)], 0)

        # The result is the maximum value in the dp array
        return max(dp)