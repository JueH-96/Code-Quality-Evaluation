from collections import defaultdict
from typing import List

class Solution:
    def countNonDecreasingSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if n == 0:
            return 0
        total = 0
        # prev_list is a list of tuples (adjusted_val, cumulative_cost, count)
        prev_list = []
        # Initialize with the first element
        prev_list.append((nums[0], 0, 1))
        total += prev_list[0][2] if prev_list[0][1] <= k else 0

        for i in range(1, n):
            curr = nums[i]
            new_list = []
            # Add the single-element subarray
            new_list.append((curr, 0, 1))
            # Extend each entry in prev_list
            for val, cost, cnt in prev_list:
                if curr >= val:
                    new_val = curr
                    new_cost = cost
                else:
                    new_val = val
                    new_cost = cost + (val - curr)
                new_list.append((new_val, new_cost, cnt))
            # Merge entries with the same (val, cost)
            merged = defaultdict(int)
            for v, c, cnt in new_list:
                merged[(v, c)] += cnt
            # Create merged_list from the merged dictionary
            merged_list = [(v, c, cnt) for (v, c), cnt in merged.items()]
            # Group by val and keep the minimum cost
            val_dict = {}
            for v, c, cnt in merged_list:
                if v not in val_dict or c < val_dict[v][0]:
                    val_dict[v] = (c, cnt)
                elif v in val_dict and c == val_dict[v][0]:
                    val_dict[v] = (c, val_dict[v][1] + cnt)
            # Create val_list sorted by val
            val_list = [(v, c, cnt) for v, (c, cnt) in val_dict.items()]
            val_list.sort()
            # Prune to keep only increasing cost
            pruned = []
            for v, c, cnt in val_list:
                if not pruned or c > pruned[-1][1]:
                    pruned.append((v, c, cnt))
            # Update prev_list for next iteration
            prev_list = pruned
            # Add valid counts to total
            for v, c, cnt in pruned:
                if c <= k:
                    total += cnt
        return total