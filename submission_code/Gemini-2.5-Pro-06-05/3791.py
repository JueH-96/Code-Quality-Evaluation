import bisect
from collections import defaultdict, deque
from typing import List

class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        n = len(fruits)
        if n == 0:
            return 0

        # Step 1: Coordinate compress basket capacities.
        # This maps large capacity values to a smaller range [0, k-1].
        unique_caps = sorted(list(set(baskets)))
        k = len(unique_caps)
        
        if k == 0:
            return n

        cap_to_comp_idx = {cap: i for i, cap in enumerate(unique_caps)}

        # Step 2: Group basket indices by their capacity.
        # Use a deque for efficient O(1) removal from the left (leftmost index).
        capacity_map = defaultdict(deque)
        for j, cap in enumerate(baskets):
            capacity_map[cap].append(j)

        # Step 3: Prepare data and build the Segment Tree.
        # The segment tree will store the minimum available index for ranges of capacities.
        min_idx_array = [capacity_map[cap][0] for cap in unique_caps]
        
        # Use n as infinity, since valid indices are 0 to n-1.
        infinity = n 
        tree = [infinity] * (4 * k)

        # Helper functions for the Segment Tree, defined within the method's scope.
        def build(node, start, end):
            if start == end:
                tree[node] = min_idx_array[start]
                return
            mid = (start + end) // 2
            left_child, right_child = 2 * node, 2 * node + 1
            build(left_child, start, mid)
            build(right_child, mid + 1, end)
            tree[node] = min(tree[left_child], tree[right_child])

        def update(node, start, end, idx, val):
            if start == end:
                tree[node] = val
                return
            mid = (start + end) // 2
            left_child, right_child = 2 * node, 2 * node + 1
            if start <= idx <= mid:
                update(left_child, start, mid, idx, val)
            else:
                update(right_child, mid + 1, end, idx, val)
            tree[node] = min(tree[left_child], tree[right_child])

        def query(node, start, end, l, r):
            if r < start or end < l:
                return infinity
            if l <= start and end <= r:
                return tree[node]
            mid = (start + end) // 2
            left_child, right_child = 2 * node, 2 * node + 1
            p1 = query(left_child, start, mid, l, r)
            p2 = query(right_child, mid + 1, end, l, r)
            return min(p1, p2)

        build(1, 0, k - 1)

        # Step 4: Process each fruit according to the rules.
        unplaced_count = 0
        for fruit in fruits:
            # Find the compressed index of the smallest capacity >= fruit.
            start_idx = bisect.bisect_left(unique_caps, fruit)
            
            # If no basket is large enough, the fruit is unplaced.
            if start_idx == k:
                unplaced_count += 1
                continue
            
            # Query for the leftmost available basket (minimum index).
            best_j = query(1, 0, k - 1, start_idx, k - 1)
            
            # If query returns infinity, no suitable basket is available.
            if best_j == infinity:
                unplaced_count += 1
                continue
                
            # A suitable basket is found. Update state for the next fruit.
            c_used = baskets[best_j]
            
            # This basket is now used, so remove its index from the queue.
            capacity_map[c_used].popleft()
            
            # Determine the new minimum index for this capacity.
            new_min_idx = infinity
            if capacity_map[c_used]:
                new_min_idx = capacity_map[c_used][0]
            
            # Update the segment tree with the new minimum index.
            comp_idx = cap_to_comp_idx[c_used]
            update(1, 0, k - 1, comp_idx, new_min_idx)
            
        return unplaced_count