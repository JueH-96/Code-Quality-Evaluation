from typing import List

class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        n = len(fruits)
        # Segment tree array, size ~4*n is sufficient
        # tree[v] stores the maximum capacity of an available basket in the range [tl, tr] covered by node v
        # Use a sentinel value like -1 to indicate an unavailable basket, ensuring it's smaller than any valid capacity (>= 1)
        tree = [0] * (4 * n)

        # Helper function to build the segment tree
        # v: current node index (1-based)
        # tl, tr: range of indices in baskets covered by this node [tl, tr]
        def build(v, tl, tr):
            if tl == tr:
                # Leaf node represents a single basket
                tree[v] = baskets[tl]
            else:
                tm = (tl + tr) // 2
                # Build left and right children
                build(2 * v, tl, tm)
                build(2 * v + 1, tm + 1, tr)
                # Internal node stores the maximum capacity in its range
                tree[v] = max(tree[2 * v], tree[2 * v + 1])

        # Helper function to find the leftmost available basket index
        # v: current node index (1-based)
        # tl, tr: range of indices in baskets covered by this node [tl, tr]
        # capacity_needed: the minimum capacity required by the current fruit
        # Returns the original index of the leftmost suitable basket, or -1 if none found
        def query(v, tl, tr, capacity_needed):
            # If the maximum capacity in this node's range is less than required,
            # no suitable basket exists in this range. Prune the search.
            if tree[v] < capacity_needed:
                return -1

            # If we reached a leaf node (single basket) and its capacity is sufficient
            # (which we know it is, otherwise the check above would have returned -1),
            # this is the leftmost suitable basket in its original index range.
            if tl == tr:
                return tl

            tm = (tl + tr) // 2

            # Prioritize searching the left child's range to find the minimum index.
            # Search left ONLY if there's a possibility of finding a suitable basket there
            # (i.e., the max capacity in the left subtree is >= capacity_needed).
            # This check on tree[2*v] is crucial for efficiency and correctness of finding min index.
            if tree[2 * v] >= capacity_needed:
                 left_result = query(2 * v, tl, tm, capacity_needed)
                 # If a suitable basket was found in the left subtree, it's the leftmost in this node's range.
                 if left_result != -1:
                     return left_result

            # If left child's range cannot contain a suitable basket (because its max capacity
            # was too low, or the recursive search in the left returned -1),
            # search the right child's range. We know tree[v] >= capacity_needed,
            # so if the left didn't contain one, one *must* be in the right (if v wasn't a leaf).
            right_result = query(2 * v + 1, tm + 1, tr, capacity_needed)
            return right_result # Returns the index found in the right, or -1

        # Helper function to mark a basket at a specific index as used (-1 capacity)
        # v: current node index (1-based)
        # tl, tr: range of indices in baskets covered by this node [tl, tr]
        # index_to_use: the original index of the basket being used
        def update(v, tl, tr, index_to_use):
            # If we reached the leaf node for the used basket
            if tl == tr:
                tree[v] = -1 # Mark as used by setting capacity to -1
            else:
                tm = (tl + tr) // 2
                # Recurse into the child covering the index_to_use
                if index_to_use <= tm:
                    update(2 * v, tl, tm, index_to_use)
                else:
                    update(2 * v + 1, tm + 1, tr, index_to_use)
                # Update the max capacity for the current node based on children
                tree[v] = max(tree[2 * v], tree[2 * v + 1])

        # Build the initial segment tree covering all baskets
        build(1, 0, n - 1)

        unplaced_count = 0
        # Process each fruit in the given order
        for fruit_quantity in fruits:
            # Find the index of the leftmost available basket that can hold this fruit
            # Search the entire range [0, n-1] starting from the root (v=1)
            basket_index = query(1, 0, n - 1, fruit_quantity)

            # If a suitable basket was found (query returned a valid index >= 0)
            if basket_index != -1:
                # Mark this basket as used in the segment tree by updating its capacity
                update(1, 0, n - 1, basket_index)
            else:
                # No suitable basket was found for this fruit
                unplaced_count += 1

        return unplaced_count