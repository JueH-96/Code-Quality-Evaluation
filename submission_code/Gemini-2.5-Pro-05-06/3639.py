import sys

class Solution:
  def isZeroArray(self, nums: list[int], queries: list[list[int]]) -> bool:
    n = len(nums)

    # Constraints: 1 <= nums.length. If n could be 0, handle as edge case.
    # if n == 0:
    #     return True 

    # Segment tree stores (min_val, max_val) for a range
    # lazy array stores values to be subtracted (always non-positive)
    st = [(0, 0)] * (4 * n) 
    lazy = [0] * (4 * n)

    def build(v: int, tl: int, tr: int):
        if tl == tr:
            st[v] = (nums[tl], nums[tl])
        else:
            tm = (tl + tr) // 2
            build(2 * v, tl, tm)
            build(2 * v + 1, tm + 1, tr)
            st[v] = (min(st[2 * v][0], st[2 * v + 1][0]), 
                     max(st[2 * v][1], st[2 * v + 1][1]))

    # tl, tr are range of node v. Needed to check if v is a leaf.
    def push(v: int, tl: int, tr: int):
        if lazy[v] == 0 or tl == tr: # No lazy tag or is a leaf node (no children to push to)
            return

        # Apply lazy tag to children
        # lazy[v] stores a negative value (or zero) representing decrements.
        
        # For child left (2*v)
        st[2*v] = (max(0, st[2*v][0] + lazy[v]), 
                   max(0, st[2*v][1] + lazy[v]))
        lazy[2*v] += lazy[v]
        
        # For child right (2*v+1)
        st[2*v+1] = (max(0, st[2*v+1][0] + lazy[v]), 
                     max(0, st[2*v+1][1] + lazy[v]))
        lazy[2*v+1] += lazy[v]
        
        lazy[v] = 0

    # v: current segment tree node index
    # tl, tr: range [tl, tr] covered by node v
    # l, r: query range [l, r]
    def update(v: int, tl: int, tr: int, l: int, r: int):
        # If current node's max value is 0, all elements in its range are 0. Nothing to decrement.
        if st[v][1] == 0:
            return
        
        # If current node's range is outside query range (no overlap)
        if tl > r or tr < l:
            return

        # If current node's range is fully within query range
        if l <= tl and tr <= r:
            # And if all elements in current node's range are positive (min_val >= 1)
            if st[v][0] >= 1:
                st[v] = (st[v][0] - 1, st[v][1] - 1)
                lazy[v] -= 1 # Mark for children to be decremented by 1 more
                return
            # Else (st[v][0] == 0 but st[v][1] > 0): 
            # Some elements are 0, some positive. Must go deeper. Fall through.
            # Note: A leaf node (tl == tr) is fully contained if l <= tl <= r.
            # If leaf_val > 0, st[v][0] >= 1 handles it.
            # If leaf_val == 0, st[v][1] == 0 handles it (first check).
            # So, if tl == tr, one of these conditions must be met, and we return.
            # Thus, code below this (push and recursion) is only for non-leaf nodes.

        # This point is reached if:
        # 1. Partial overlap.
        # 2. Full overlap, but st[v][0] == 0 (and st[v][1] > 0), so mixed values.
        # Both require push and recursion. We know tl != tr (not a leaf).
        push(v, tl, tr) 

        tm = (tl + tr) // 2
        update(2 * v, tl, tm, l, r)
        update(2 * v + 1, tm + 1, tr, l, r)
        
        # After children are updated, recompute st[v]
        st[v] = (min(st[2*v][0], st[2*v+1][0]), 
                 max(st[2*v][1], st[2*v+1][1]))

    build(1, 0, n - 1)

    for q_l, q_r in queries:
        # Optimization: if all numbers are already zero, no query can change them.
        if st[1][1] == 0: # Max element in entire array is 0
            break 
        update(1, 0, n - 1, q_l, q_r)
        
    # After all queries, check if all elements are zero.
    # This means the max element in the entire array (root of segment tree) must be 0.
    return st[1][1] == 0