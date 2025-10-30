from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        # Compute smallest prime factor for all numbers up to 10^6
        max_num = 1000000
        spf = [0] * (max_num + 1)
        for i in range(2, max_num + 1):
            if spf[i] == 0:
                for j in range(i, max_num + 1, i):
                    if spf[j] == 0:
                        spf[j] = i
        # Define INF
        INF = 10**9 + 5
        # Handle position 0
        num0 = nums[0]
        if num0 == 1 or spf[num0] == num0:  # 1 or prime
            prev_dp = [(num0, 0)]  # (value, total_cost)
        else:  # composite
            spf_val = spf[num0]
            prev_dp = [(num0, 0), (spf_val, 1)]
        # Now for each subsequent position from 1 to n-1
        for i in range(1, n):
            num_curr = nums[i]
            # Get current choices: list of (val, cost_add)
            if num_curr == 1 or spf[num_curr] == num_curr:  # 1 or prime
                curr_choices = [(num_curr, 0)]
            else:  # composite
                spf_val_curr = spf[num_curr]
                curr_choices = [(num_curr, 0), (spf_val_curr, 1)]
            # Compute curr_dp: list of (val, dp_cost)
            curr_dp_list = []
            has_finite_cost = False
            for curr_val, cost_add_curr in curr_choices:
                # Find min prev cost where prev_val <= curr_val
                min_prev_cost = INF
                for prev_val, prev_cost in prev_dp:
                    if prev_val <= curr_val and prev_cost < INF:
                        if prev_cost < min_prev_cost:
                            min_prev_cost = prev_cost
                if min_prev_cost == INF:
                    dp_cost_curr = INF  # cannot reach
                else:
                    dp_cost_curr = cost_add_curr + min_prev_cost
                curr_dp_list.append((curr_val, dp_cost_curr))
                if dp_cost_curr < INF:
                    has_finite_cost = True
            if not has_finite_cost:
                return -1  # impossible to reach this position
            # Set prev_dp to curr_dp_list
            prev_dp = curr_dp_list
        # After all positions, find min cost in prev_dp
        min_cost = INF
        for _, dp_cost in prev_dp:
            if dp_cost < min_cost:
                min_cost = dp_cost
        return min_cost