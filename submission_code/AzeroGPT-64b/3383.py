from decimal import Decimal

class Solution:
    def maximumEnergy(self, energy, k):
        n = len(energy)
        presum = [0] * (n + 1)
        for i in range(1, n + 1):
            presum[i] = presum[i - 1] + energy[(i - 1) * k % n]

        values = [presum[i * k % n] - presum[(i - 1) * k % n] for i in range(n)]
        values.sort(reverse=True)
        current_sum = 0
        max_sum = Decimal('-inf')
        dp = [Decimal('-inf')] * len(values)
        for i in range(len(values)):
            current_sum = max(current_sum + values[i], Decimal(0))
            max_sum = max(max_sum, current_sum)
            dp[i] = max_sum
        return int(dp[len(values) - 1])