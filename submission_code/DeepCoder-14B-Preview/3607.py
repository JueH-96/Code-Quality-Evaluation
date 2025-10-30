import math
from bisect import bisect_right

class Solution:
    def minOperations(self, nums: list) -> int:
        def get_reachable_values(x):
            if x == 1:
                return [(1, 0)]
            reachable = []
            current = x
            steps = 0
            reachable.append((current, steps))
            while True:
                # Find the smallest prime factor
                spf = None
                for i in range(2, int(math.sqrt(current)) + 1):
                    if current % i == 0:
                        spf = i
                        break
                if spf is None:
                    # current is a prime, can't be reduced further
                    break
                else:
                    gpd = current // spf
                    if gpd == 1:
                        break
                    current = current // gpd
                    steps += 1
                    reachable.append((current, steps))
            return reachable
        
        if not nums:
            return 0
        
        reachable_list = []
        for x in nums:
            reachable = get_reachable_values(x)
            if not reachable:
                return -1
            reachable_list.append(reachable)
        
        # Initialize DP for the first element
        first_reachable = reachable_list[0]
        first_sorted = sorted(first_reachable, key=lambda x: x[0])
        dp = first_sorted
        
        for i in range(1, len(nums)):
            current_reachable = reachable_list[i]
            if not current_reachable:
                return -1
            
            current_sorted = sorted(current_reachable, key=lambda x: x[0])
            
            # Prepare previous dp for binary search
            prev_sorted = sorted(dp, key=lambda x: x[0])
            v_prev_list = [x[0] for x in prev_sorted]
            steps_prev_list = [x[1] for x in prev_sorted]
            
            # Compute prefix minima
            prefix_min = []
            min_so_far = float('inf')
            for s in steps_prev_list:
                if s < min_so_far:
                    min_so_far = s
                prefix_min.append(min_so_far)
            
            current_dp = []
            for (v_curr, steps_curr) in current_sorted:
                idx = bisect_right(v_prev_list, v_curr) - 1
                if idx >= 0:
                    minimal_prev = prefix_min[idx]
                    total_steps = minimal_prev + steps_curr
                    current_dp.append((v_curr, total_steps))
            
            if not current_dp:
                return -1
            
            # Update dp for the next iteration
            current_dp_sorted = sorted(current_dp, key=lambda x: x[0])
            dp = current_dp_sorted
        
        if not dp:
            return -1
        min_steps = min([s for (v, s) in dp])
        return min_steps