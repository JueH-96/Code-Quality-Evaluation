import collections
import heapq # Not strictly needed for this DSU solution, but common for greedy problems.

class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        n = len(maximumHeight)

        # DSU structure to find the largest available height slot <= k
        # parent[i] stores the largest available height h such that h <= i.
        # When a height h is assigned, we mark it as used by setting parent[h]
        # to the representative of (h - 1). This effectively "removes" h
        # from the set of available heights and makes future find(h') calls (for h' >= h)
        # resolve to a smaller available height.
        
        # Use a dictionary for parent to handle large and sparse height values (up to 10^9).
        # A list/array is not feasible for such a large range.
        parent = {}

        def find(i):
            # If 'i' is 0, it means we've exhausted all positive heights.
            if i <= 0:
                return 0
            
            # If 'i' is not in 'parent', it means 'i' hasn't been part of any DSU operation yet.
            # In this case, 'i' itself is available, and it's its own root/representative.
            # Also, if parent[i] == i, it's a root.
            if i not in parent or parent[i] == i:
                parent[i] = i
                return i
            
            # Path compression: make the current node point directly to its root.
            # This optimizes future lookups for elements in this path.
            parent[i] = find(parent[i])
            return parent[i]
        
        total_sum = 0
        
        # Sort maximumHeight in descending order.
        # This greedy strategy prioritizes assigning larger heights to towers
        # that have higher maximum limits. This ensures that smaller heights
        # are left available for towers with tighter restrictions, which helps
        # maximize the overall sum while satisfying distinctness.
        maximumHeight.sort(reverse=True)

        # Iterate through the towers in decreasing order of their maximum height limits.
        for max_h_limit in maximumHeight:
            # Find the largest available height slot (`actual_h`) that is
            # less than or equal to the current `max_h_limit`.
            actual_h = find(max_h_limit)
            
            # If `actual_h` is 0, it means no positive height (1 or greater) is available
            # that satisfies the `max_h_limit` constraint. This implies it's impossible
            # to assign a distinct positive height to the current tower.
            # In this scenario, no valid assignment exists for all towers.
            if actual_h <= 0:
                return -1 
            
            # Assign `actual_h` to the current tower and add it to the total sum.
            total_sum += actual_h
            
            # Mark `actual_h` as used by updating its parent in the DSU.
            # By setting parent[actual_h] to find(actual_h - 1), any future attempt
            # to find a slot that was previously `actual_h` will now resolve to
            # the next available smaller slot.
            parent[actual_h] = find(actual_h - 1)
            
        return total_sum