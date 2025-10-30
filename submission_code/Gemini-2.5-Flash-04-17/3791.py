from typing import List
import math

class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        n = len(fruits)
        INF = float('inf') 

        # Segment Tree Implementation
        # tree array size is approximately 4*N for 1-based indexing
        tree = [0] * (4 * n)
        initial_basket_capacities = baskets

        # Build function to initialize the segment tree
        # v: current node index
        # tl, tr: range [tl, tr] covered by node v
        # Stores min capacity of available baskets in range [tl, tr]
        def build(v, tl, tr):
            if tl == tr:
                tree[v] = initial_basket_capacities[tl]
            else:
                tm = (tl + tr) // 2
                build(v * 2, tl, tm)
                build(v * 2 + 1, tm + 1, tr)
                tree[v] = min(tree[v * 2], tree[v * 2 + 1])

        # Update function to mark a basket as used (set its value to INF)
        # v: current node index
        # tl, tr: range [tl, tr] covered by node v
        # pos: index of the basket to update
        # new_val: value to set (INF)
        def update(v, tl, tr, pos, new_val):
            if tl == tr:
                tree[v] = new_val
            else:
                tm = (tl + tr) // 2
                if pos <= tm:
                    update(v * 2, tl, tm, pos, new_val)
                else:
                    update(v * 2 + 1, tm + 1, tr, pos, new_val)
                tree[v] = min(tree[v * 2], tree[v * 2 + 1])

        # Search function to find the leftmost available basket index that fits the fruit
        # v: current node index
        # tl, tr: range [tl, tr] covered by node v
        # fruit_size: required capacity
        # Returns the index of the first fitting available basket or -1 if none found
        def find_first_fitting_index(v, tl, tr, fruit_size):
            # If the minimum capacity in this range is less than required,
            # no available basket in this range can fit the fruit.
            # This check prunes entire subtrees.
            # This also covers the case where the range contains only INF values (all baskets used).
            if tree[v] < fruit_size:
                 return -1

            # If this is a leaf node, and its capacity is sufficient (checked above),
            # this is the leftmost available fitting basket in this trivial range.
            if tl == tr:
                return tl

            tm = (tl + tr) // 2

            # Prioritize searching the left child range [tl, tm]
            # If the minimum capacity in the left child range [tl, tm] is sufficient,
            # it implies there is at least one basket that fits in the left range.
            # The leftmost fitting basket overall *might* be in this left range,
            # so we must search here first.
            # If tree[v*2] < fruit_size, then the leftmost cannot be in the left child range.
            if tree[v * 2] >= fruit_size:
                 left_result = find_first_fitting_index(v * 2, tl, tm, fruit_size)
                 if left_result != -1:
                     return left_result

            # If no fitting basket was found in the left child range (either because
            # tree[v*2] < fruit_size or the recursive call returned -1), search the right child range [tm+1, tr].
            # Check if the right child range *could* contain a fitting basket.
            # If the minimum capacity in the right child range [tm+1, tr] is sufficient,
            # it implies there is at least one basket that fits in the right range.
            # The leftmost fitting basket overall *might* be in this right range
            # (since none was found in the left).
            if tree[v * 2 + 1] >= fruit_size:
                 right_result = find_first_fitting_index(v * 2 + 1, tm + 1, tr, fruit_size)
                 if right_result != -1:
                     return right_result

            # If neither child search found a fitting basket
            return -1


        # Build the initial segment tree
        build(1, 0, n - 1)

        unplaced_count = 0
        for fruit_size in fruits:
            # Find the leftmost available basket that fits the current fruit
            found_index = find_first_fitting_index(1, 0, n - 1, fruit_size)

            if found_index != -1:
                # Basket found, mark it as used by setting its capacity to INF in the tree
                update(1, 0, n - 1, found_index, INF)
            else:
                # No available basket found for this fruit
                unplaced_count += 1

        return unplaced_count