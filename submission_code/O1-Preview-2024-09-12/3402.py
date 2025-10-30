class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        from collections import Counter
        mod = 10 ** 9 + 7
        counts_needed = []
        max_increment = 0
        for num in nums:
            inc = max(nums) - num
            if inc > 0:
                counts_needed.append(inc)
                max_increment = max(max_increment, inc)
            else:
                counts_needed.append(0)
        if cost2 >= 2 * cost1:
            # It's cheaper or equal to use op1 for all increments
            total_cost = sum(counts_needed) * cost1
            return total_cost % mod
        # Build counts_needed per increments_needed
        counts_counter = Counter(counts_needed)
        increments_list = sorted([inc for inc in counts_counter.keys() if inc > 0], reverse=True)
        total_cost = 0
        n_t = 0  # number of elements needing increments at current level
        previous_increments_needed = 0
        for increments_needed in increments_list + [0]:
            delta_steps = increments_needed - previous_increments_needed
            n_t += counts_counter[increments_needed]
            previous_increments_needed = increments_needed
            if delta_steps == 0:
                continue
            steps = delta_steps
            per_step_cost = 0
            if cost2 >= 2 * cost1:
                per_step_cost = n_t * cost1
            else:
                pairs = n_t // 2
                singles = n_t % 2
                per_step_cost = pairs * cost2 + singles * cost1
            total_cost += steps * per_step_cost
        return total_cost % mod