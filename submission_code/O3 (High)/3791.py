from typing import List

class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        """
        For every fruit (visited from left to right) we must find the left-most
        still free basket whose capacity is at least the required amount.
        After using a basket it becomes unavailable.

        We build an index-based segment tree that stores, for every segment,
        the maximum capacity of the still unused baskets in this segment.
        •  Query:  “Is there any basket with capacity ≥ x ?”  → root max
        •  Search: “Left-most index whose capacity ≥ x”       → walk the tree
        •  Update: mark a basket as used (capacity = 0)

        Each operation costs O(log n); with n ≤ 1e5 this easily fits.
        """
        n = len(fruits)
        if n == 0:
            return 0

        # --- build a power-of-two sized segment tree -------------------------
        size = 1
        while size < n:
            size <<= 1

        tree = [0] * (2 * size)                       # leaves start at index `size`
        for i in range(n):
            tree[size + i] = baskets[i]
        for i in range(size - 1, 0, -1):              # build internal nodes
            tree[i] = max(tree[i << 1], tree[i << 1 | 1])

        # --- helper: first (left-most) index with capacity ≥ need ------------
        def first_idx_at_least(need: int) -> int:     # -1 if none exists
            if tree[1] < need:                        # root max too small
                return -1
            idx, l, r = 1, 0, size - 1
            while idx < size:                         # while not at leaf
                mid = (l + r) // 2
                if tree[idx << 1] >= need:            # enough capacity in left child
                    idx = idx << 1
                    r = mid
                else:                                 # must go right
                    idx = idx << 1 | 1
                    l = mid + 1
            return l                                  # `l` is the leaf's real index

        # --- helper: mark basket `pos` as used --------------------------------
        def use_basket(pos: int) -> None:
            p = size + pos
            tree[p] = 0                               # capacity 0 ⇒ unusable
            p >>= 1
            while p:
                tree[p] = max(tree[p << 1], tree[p << 1 | 1])
                p >>= 1

        # --- main loop --------------------------------------------------------
        unplaced = 0
        for need in fruits:
            idx = first_idx_at_least(need)
            if idx == -1 or idx >= n:                 # no suitable basket
                unplaced += 1
            else:
                use_basket(idx)

        return unplaced