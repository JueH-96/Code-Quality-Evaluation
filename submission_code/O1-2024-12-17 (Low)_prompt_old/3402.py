class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        """
        We want all elements to become equal using:
          - Increase one element by 1 at a cost of cost1.
          - Increase two distinct elements by 1 each at a cost of cost2.

        Key observations:
        1) Since we can only increase, the final value must be at least max(nums).
        2) Allowing any element already at max(nums) to be increased further in hopes
           of pairing increments is never cheaper overall, because we would then pay
           for increments we do not really need on larger elements.
           Thus, the optimal final target is T = max(nums).
        3) Let deficits[i] = T - nums[i]. The total increments needed is sum(deficits).
           We can either:
             (a) pay cost1 per increment, or
             (b) pair up increments among elements still below T if cost2 < 2*cost1.
        4) However, pairing is only useful if cost2 < 2 * cost1, and we can only pair
           as long as at least two elements still need increments. A closed-form way
           to compute how many increments can be paired is:
             - sum_of_deficits = sum(deficits).
             - max_deficit = max(deficits).
             - leftover = 2 * max_deficit - sum_of_deficits
               (this tells us how many increments of the largest deficit cannot be paired
                once the other deficits are exhausted).
             - leftover_count = max(leftover, 0) plus possibly 1 more if the total needed
               is odd (to account for an unpaired increment).
             - pair_count = (sum_of_deficits - leftover_count) // 2
             - cost_pair = pair_count*cost2 + leftover_count*cost1
           We then compare cost_pair with the cost of doing everything singly (cost_single),
           and take the minimum.

        Finally, return the result modulo 10^9 + 7.
        """

        MOD = 10**9 + 7
        if not nums:
            return 0
        
        # T is the required final value (the maximum of nums).
        T = max(nums)

        # Calculate the total number of increments needed and track the largest deficit.
        sum_of_deficits = 0
        max_deficit = 0
        for x in nums:
            diff = T - x
            sum_of_deficits += diff
            if diff > max_deficit:
                max_deficit = diff
        
        # If sum_of_deficits is 0, array is already equal; cost is 0.
        if sum_of_deficits == 0:
            return 0
        
        # Cost if we do all single increments:
        cost_single = sum_of_deficits * cost1
        
        # Compute the pairing strategy cost:
        # leftover is how many increments of the largest deficit cannot be paired
        # once the other deficits are exhausted.
        leftover = 2 * max_deficit - sum_of_deficits
        if leftover < 0:
            leftover = 0
        
        # We also need to account for odd/even total increments.
        # If (sum_of_deficits - leftover) is odd, one extra single increment is needed.
        leftover += (sum_of_deficits - leftover) % 2
        
        pair_count = (sum_of_deficits - leftover) // 2
        cost_pair = pair_count * cost2 + leftover * cost1
        
        # Final answer is the cheaper of the two approaches.
        ans = min(cost_single, cost_pair) % MOD
        return ans