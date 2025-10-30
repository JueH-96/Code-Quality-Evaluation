from typing import List
from bisect import bisect_left

class Fenw:
    # Fenwick tree for range maximum queries
    def __init__(self, n: int):
        self.n = n
        self.data = [-1] * (n + 1)
    
    def update(self, i: int, val: int):
        # i is 0-indexed for our usage but tree is 1-indexed internally.
        i += 1
        while i <= self.n:
            self.data[i] = max(self.data[i], val)
            i += i & -i
    
    def query(self, i: int) -> int:
        # query the maximum value in the range [0, i], where i is 0-indexed.
        res = -1
        i += 1
        while i:
            res = max(res, self.data[i])
            i -= i & -i
        return res

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        
        # Prepare a list of indices for elements in the arrays
        # We will sort by nums1 descending order.
        arr = [(nums1[i], nums2[i], nums1[i] + nums2[i]) for i in range(n)]
        arr.sort(key=lambda x: x[0], reverse=True)
        
        # Prepare queries: store with original index
        # Each query is of the form [x, y]
        q_with_idx = [(q[0], q[1], i) for i, q in enumerate(queries)]
        q_with_idx.sort(key=lambda x: x[0], reverse=True)
        
        # Coordinate compression for nums2 values from the original array and queries
        # We need to compress potential nums2 values that appear in the data for Fenw queries
        all_nums2 = []
        for a, b, _ in arr:
            all_nums2.append(b)
        for _, y, _ in q_with_idx:
            all_nums2.append(y)
        all_nums2 = sorted(set(all_nums2))
        
        # mapping function: value -> index in Fenw tree, lower bound
        def get_index(y: int) -> int:
            return bisect_left(all_nums2, y)
        
        size = len(all_nums2)
        fenw = Fenw(size)
        
        res = [-1] * len(queries)
        j = 0  # pointer for arr list
        
        # Process queries in descending order of x
        for x_req, y_req, idx in q_with_idx:
            # Add all array elements with nums1 >= x_req into the fenw tree.
            while j < len(arr) and arr[j][0] >= x_req:
                _, b_val, sum_val = arr[j]
                pos = get_index(b_val)
                fenw.update(pos, sum_val)
                j += 1
            # Query fenw tree: we need to get maximum among all nums2 >= y_req.
            pos_query = get_index(y_req)
            max_val = -1
            # To get maximum among indices >= pos_query, we can maintain an auxiliary structure,
            # but we can simulate this by doing a Fenw query over prefix up to last index and then subtracting
            # the prefix up to pos_query - 1. However, since our Fenw tree is built for prefix max, we can't directly
            # query suffix maximum. So instead, we can build the Fenw tree so that index order is reversed:
            # i.e., use reversed order for nums2. We'll rebuild using that approach.
            # Instead, we will re-interpret the Fenw tree: We'll build tree where the index represents reversed
            # order such that larger nums2 give smaller index. So reinitialize.
            # However, easier is to use a segment tree approach, but since constraints are large.

            # Alternative approach: Use Fenw tree for prefix maximum if we reverse the coordinate order.
            # Let's implement a separate Fenw tree for "reversed" coordinates concurrently.
            # I will try a different approach: Instead of using fenw query as prefix maximum,
            # we'll store all elements by reversed index order.
            # We'll build a new fenw tree using the following trick:
            # Let pos_rev = size - 1 - pos. Then we update pos_rev position with sum_val.
            # Then query on [get_index_rev(y_req), size-1] is prefix query on reversed tree.
            # We'll build a new fenw tree for reversed order.

            # So instead of above, we build a second fenw tree and update accordingly.
            pass

# Revised solution using reversed Fenw tree for suffix maximum query.
from typing import List
from bisect import bisect_left

class FenwRev:
    # Fenw tree for maximum queries, but we will store the reversed order such that a query over [0, pos] in the tree
    # corresponds to a query over [pos_rev, end] in the original ordering.
    def __init__(self, n: int):
        self.n = n
        self.data = [-1] * (n + 1)
    
    def update(self, i: int, val: int):
        # i is 0-indexed in our reversed tree.
        i += 1
        while i <= self.n:
            self.data[i] = max(self.data[i], val)
            i += i & -i
    
    def query(self, i: int) -> int:
        # query maximum in range [0, i]
        res = -1
        i += 1
        while i:
            res = max(res, self.data[i])
            i -= i & -i
        return res

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        # Prepare array elements: (nums1, nums2, sum)
        arr = [(nums1[i], nums2[i], nums1[i] + nums2[i]) for i in range(n)]
        arr.sort(key=lambda x: x[0], reverse=True)  # sort descending by nums1
        
        # Prepare queries with original indices, queries is [x, y]
        q_with_idx = [(q[0], q[1], i) for i, q in enumerate(queries)]
        q_with_idx.sort(key=lambda x: x[0], reverse=True)  # sort descending by x
        
        # Coordinate compression for nums2 (from array and queries)
        all_nums2 = []
        for _, b, _ in arr:
            all_nums2.append(b)
        for _, y, _ in q_with_idx:
            all_nums2.append(y)
        all_nums2 = sorted(set(all_nums2))
        size = len(all_nums2)
        
        # For reversed Fenw, we want to query for all elements with nums2 >= y.
        # Let reverse index be: rev_index = size - 1 - original_index.
        def get_rev_index(val: int) -> int:
            # find original index for val using lower bound and then convert.
            orig_index = bisect_left(all_nums2, val)
            return size - 1 - orig_index
        
        fenw_rev = FenwRev(size)
        res = [-1] * len(queries)
        j = 0
        
        for x_req, y_req, q_idx in q_with_idx:
            # add all array elements with nums1 >= x_req
            while j < len(arr) and arr[j][0] >= x_req:
                _, b_val, sum_val = arr[j]
                pos_rev = get_rev_index(b_val)
                fenw_rev.update(pos_rev, sum_val)
                j += 1
            # query for maximum among elements with nums2 >= y_req
            pos_rev_query = get_rev_index(y_req)
            max_val = fenw_rev.query(pos_rev_query)
            res[q_idx] = max_val
        return res