import math
from functools import reduce
from typing import List

class Solution:
    def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
        m = len(target)
        
        def gen_partitions(indices):
            if not indices:
                yield frozenset()
                return
            first = indices[0]
            rest = indices[1:]
            for part_rest in gen_partitions(rest):
                # Add as new group
                new_part1 = part_rest | frozenset([frozenset([first])])
                yield new_part1
                # Add to each existing group
                for group in list(part_rest):
                    new_group = group.union(frozenset([first]))
                    new_part_mod = (part_rest - frozenset([group])) | frozenset([new_group])
                    yield new_part_mod
        
        def lcm_multiple(vals):
            return reduce(lambda x, y: x * y // math.gcd(x, y), vals)
        
        ans = float('inf')
        
        for partition in gen_partitions(tuple(range(m))):
            groups = list(partition)  # list of frozensets of indices
            k = len(groups)
            L_list = []
            for group in groups:
                subset_indices = sorted(list(group))  # sort indices
                target_vals = [target[idx] for idx in subset_indices]
                lcm_val = lcm_multiple(target_vals)
                L_list.append(lcm_val)
            
            # Now DP
            dp_prev = [float('inf')] * (1 << k)
            dp_prev[0] = 0
            
            for num_val in nums:
                dp_curr = [float('inf') for _ in range(1 << k)]
                for mask in range(1 << k):
                    # Not use num
                    dp_curr[mask] = min(dp_curr[mask], dp_prev[mask])
                    # Use num for each j if j in mask
                    for j in range(k):
                        if (mask & (1 << j)) != 0:
                            prev_mask = mask ^ (1 << j)
                            cost_add = 0 if num_val % L_list[j] == 0 else L_list[j] - (num_val % L_list[j])
                            dp_curr[mask] = min(dp_curr[mask], dp_prev[prev_mask] + cost_add)
                dp_prev = dp_curr  # assign
            
            cost_partition = dp_prev[(1 << k) - 1]
            ans = min(ans, cost_partition)
        
        return int(ans)