import sys

class SegmentTree:
    def __init__(self, size, initial_values):
        self.size = size
        self.tree = [0] * (4 * size)
        self.lazy = [0] * (4 * size)
        if initial_values:
            self._build(1, 0, size - 1, initial_values)

    def _build(self, v, tl, tr, initial_values):
        if tl == tr:
            self.tree[v] = initial_values[tl]
        else:
            tm = (tl + tr) // 2
            self._build(v * 2, tl, tm, initial_values)
            self._build(v * 2 + 1, tm + 1, tr, initial_values)
            self.tree[v] = max(self.tree[v * 2], self.tree[v * 2 + 1])

    def _push(self, v):
        if self.lazy[v] != 0:
            self.tree[v * 2] += self.lazy[v]
            self.lazy[v * 2] += self.lazy[v]
            self.tree[v * 2 + 1] += self.lazy[v]
            self.lazy[v * 2 + 1] += self.lazy[v]
            self.lazy[v] = 0

    def update_range(self, v, tl, tr, l, r, add):
        if l > r:
            return
        if l == tl and r == tr:
            self.tree[v] += add
            self.lazy[v] += add
        else:
            self._push(v)
            tm = (tl + tr) // 2
            self.update_range(v * 2, tl, tm, l, min(r, tm), add)
            self.update_range(v * 2 + 1, tm + 1, tr, max(l, tm + 1), r, add)
            self.tree[v] = max(self.tree[v * 2], self.tree[v * 2 + 1])

    def query_max(self, v, tl, tr, l, r):
        if l > r:
            return -sys.maxsize # Represents negative infinity
        if l == tl and r == tr:
            return self.tree[v]
        self._push(v)
        tm = (tl + tr) // 2
        return max(self.query_max(v * 2, tl, tm, l, min(r, tm)),
                   self.query_max(v * 2 + 1, tm + 1, tr, max(l, tm + 1), r))

    def updateRange(self, l, r, add):
        if self.size == 0: return
        self.update_range(1, 0, self.size - 1, l, r, add)

    def queryMax(self, l, r):
        if self.size == 0 or l > r: return -sys.maxsize
        return self.query_max(1, 0, self.size - 1, l, r)

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Precompute DP_prefix: distinct count in A[0...i]
    distinct_prefix = [0] * N
    seen = set()
    for i in range(N):
        seen.add(A[i])
        distinct_prefix[i] = len(seen)

    # Precompute DP_suffix: distinct count in A[k...N-1]
    distinct_suffix = [0] * N
    seen = set()
    for i in range(N - 1, -1, -1):
        seen.add(A[i])
        distinct_suffix[i] = len(seen)

    # Segment tree will operate on indices i from 0 to N-3
    # These are the possible end indices of the first subarray
    seg_tree_size = N - 2
    if seg_tree_size <= 0: # N must be >= 3 based on constraints
         pass # Should not happen for N >= 3

    initial_seg_values = []
    if seg_tree_size > 0:
        initial_seg_values = [distinct_prefix[i] for i in range(seg_tree_size)]

    seg_tree = SegmentTree(seg_tree_size, initial_seg_values)

    last_pos = {} # Map value to its last seen index
    max_total_sum = 0

    # Iterate k from 0 to N-1. A[k] is the current element being processed.
    # When processing A[k], this element is the rightmost element
    # for all potential second subarrays ending at index k.
    # A[i+1 ... k]. The first cut is at i, 0 <= i < k.
    # The second cut is at k.
    for k in range(N):
        # A[k] being added to the right end of ranges A[i+1 ... k-1] to form A[i+1 ... k].
        # v = A[k]. p is the last occurrence of v before index k.
        v = A[k]
        p = last_pos.get(v, -1)

        # For each i in [0, k-1], the distinct count D(i+1, k)
        # is D(i+1, k-1) + 1 if v did not appear in A[i+1 ... k-1].
        # v did not appear in A[i+1 ... k-1] iff the last occurrence of v before k is at index <= i.
        # This means i >= p.
        # So, for i in [max(0, p), k-1], the value DP_prefix[i] + D(i+1, k) is 1 greater
        # than DP_prefix[i] + D(i+1, k-1).
        # The segment tree at index i should store DP_prefix[i] + D(i+1, k-1) before this step.
        # After processing A[k], we effectively update these values.
        # Range update +1 on [max(0, p), k-1].
        # Segment tree indices i run from 0 to N-3.
        # The update range for i is intersection of [max(0, p), k-1] and [0, N-3].
        update_range_start = max(0, p)
        update_range_end = k - 1
        
        if seg_tree_size > 0 and update_range_start <= update_range_end:
            seg_tree.updateRange(update_range_start, min(update_range_end, seg_tree_size - 1), 1)

        # If k is a valid index for the second cut (i.e., end of the second subarray),
        # which means the first cut is at i (0 <= i <= k-1) and the third subarray starts at k+1 (k+1 <= N-1).
        # The condition 0 <= i < k and k < N-1 is equivalent to 0 <= i <= k-1 and 1 <= k <= N-2.
        # So, if 1 <= k <= N-2, k is a possible end index for the second subarray.
        if 1 <= k <= N - 2:
            # The second cut is after index k.
            # The first cut is after index i, where 0 <= i <= k-1.
            # We want to maximize D(0, i) + D(i+1, k) + D(k+1, N-1) over i in [0, k-1].
            # This is DP_prefix[i] + D(i+1, k) + DP_suffix[k+1].
            # DP_suffix[k+1] is fixed for this k.
            # We need max(DP_prefix[i] + D(i+1, k)) over i in [0, k-1].
            # After the range update step above, the segment tree at index i (for 0 <= i <= k-1 and i <= N-3)
            # effectively stores DP_prefix[i] + D(i+1, k).
            # Query max in range [0, k-1].
            query_range_start = 0
            query_range_end = k - 1

            if seg_tree_size > 0 and query_range_start <= query_range_end:
                U_k = seg_tree.queryMax(query_range_start, query_range_end)
                max_total_sum = max(max_total_sum, U_k + distinct_suffix[k + 1])

        # Update last_pos for the current element A[k].
        last_pos[v] = k

    print(max_total_sum)


solve()