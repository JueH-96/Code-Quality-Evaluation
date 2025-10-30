class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        """
        We want to make all elements equal to max(nums) using two types of operations:
          1) Increment one element by 1 at cost = cost1.
          2) Increment two distinct elements by 1 each at cost = cost2.

        If cost2 > 2*cost1, it is never beneficial to use the second operation
        (because two single increments cost 2*cost1, which is cheaper). So we just do
        all increments singly.

        Otherwise, we first compute totalD = sum(maxVal - nums[i]) where maxVal = max(nums).
        Then we can pair up as many increments as possible, but we are limited by:
          - floor(totalD / 2)
          - and totalD - (maxVal - minVal)
            (because once one element has used up all others for pairing, the remainder
             must be done singly).
        Let mp = min(floor(totalD/2), totalD - (maxVal - minVal)).
        The cost is then:
          mp * cost2 + (totalD - 2*mp) * cost1

        Finally, return this cost modulo 10^9+7.
        """
        MOD = 10**9 + 7
        max_val = max(nums)
        min_val = min(nums)

        # total increments needed to bring all up to max_val
        totalD = sum(max_val - x for x in nums)

        # If pairing is more expensive than 2 single increments, just do singles
        if cost2 > 2 * cost1:
            return (totalD % MOD) * (cost1 % MOD) % MOD

        # Otherwise, use as many pairs as possible
        M = max_val - min_val  # largest delta among all elements
        # Max possible pairs
        max_pairs = min(totalD // 2, totalD - M)

        # Compute the cost
        cost = (cost2 * max_pairs) + cost1 * (totalD - 2 * max_pairs)
        return cost % MOD