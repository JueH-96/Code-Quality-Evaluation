import math

# Segment Tree class for Range Minimum Queries
class SegmentTree:
    def __init__(self, arr: list[float], INF: float):
        self.n = len(arr)
        self.INF = INF
        if self.n == 0: # Handle empty input array case
            self.tree = []
            return
            
        self.tree = [INF] * (4 * self.n)
        self.arr = arr # Keep a reference if needed, or copy
        self._build(0, 0, self.n - 1)

    def _build(self, node_idx: int, start: int, end: int):
        if start == end:
            self.tree[node_idx] = self.arr[start]
        else:
            mid = (start + end) // 2
            left_child = 2 * node_idx + 1
            right_child = 2 * node_idx + 2
            self._build(left_child, start, mid)
            self._build(right_child, mid + 1, end)
            self.tree[node_idx] = min(self.tree[left_child], self.tree[right_child])

    def _query(self, node_idx: int, start: int, end: int, l: int, r: int) -> float:
        if r < start or end < l or l > r: # Query range is outside or invalid
            return self.INF
        if l <= start and end <= r: # Current segment is completely within query range
            return self.tree[node_idx]
        
        mid = (start + end) // 2
        left_child = 2 * node_idx + 1
        right_child = 2 * node_idx + 2
        p1 = self._query(left_child, start, mid, l, r)
        p2 = self._query(right_child, mid + 1, end, l, r)
        return min(p1, p2)

    def query_range(self, l: int, r: int) -> float:
        if self.n == 0 or l > r: # Handle query on empty tree or invalid range
            return self.INF
        return self._query(0, 0, self.n - 1, l, r)

class Solution:
  def minimumValueSum(self, nums: list[int], andValues: list[int]) -> int:
    n = len(nums)
    m = len(andValues)
    
    INF = float('inf')
    MAX_AND_VAL = (1 << 18) - 1 # All 1s for bit range up to 10^5 (< 2^17)

    dp = [[INF] * n for _ in range(m)]

    # Base case: i = 0 (first group)
    current_AND_prefix = MAX_AND_VAL
    for j in range(n):
        current_AND_prefix &= nums[j]
        if current_AND_prefix == andValues[0]:
            dp[0][j] = nums[j]

    # Inductive step: i from 1 to m-1
    for i in range(1, m):
        st = SegmentTree(dp[i-1], INF)
        
        # j is end index of current group (group i+1, using andValues[i])
        # Min j is i (i+1 groups need at least i+1 elements nums[0...i])
        for j in range(i, n):
            current_segment_AND = MAX_AND_VAL
            # p_seg_right_boundary: for start indices p in [k, p_seg_right_boundary], AND(nums[p..j]) == current_segment_AND
            p_seg_right_boundary = j 
            
            # k_loop: potential start index p for current group nums[p..j]
            # Iterate k_loop from j down to i. (p-1 >= i-1 => p >= i)
            for k_loop in range(j, i - 1, -1): 
                new_AND = current_segment_AND & nums[k_loop]

                if new_AND != current_segment_AND:
                    # AND value changed. current_segment_AND was for p in [k_loop + 1, p_seg_right_boundary].
                    if current_segment_AND == andValues[i]:
                        # Prev group ended at indices x in [k_loop, p_seg_right_boundary - 1].
                        # Smallest index for dp[i-1][x] is i-1.
                        query_idx_start = max(i - 1, k_loop)
                        query_idx_end = p_seg_right_boundary - 1
                        
                        min_prev_sum = st.query_range(query_idx_start, query_idx_end)
                        if min_prev_sum != INF:
                            dp[i][j] = min(dp[i][j], min_prev_sum + nums[j])
                    
                    current_segment_AND = new_AND
                    p_seg_right_boundary = k_loop # This k_loop starts a new segment

                # Optimization: if current_segment_AND has a 0 where andValues[i] has a 1.
                if (current_segment_AND | andValues[i]) != andValues[i]:
                    break 
                
                # Check current segment of start indices p in [k_loop, p_seg_right_boundary]. All yield current_segment_AND.
                if current_segment_AND == andValues[i]:
                    # Prev group ended at indices x in [k_loop - 1, p_seg_right_boundary - 1].
                    query_idx_start = max(i - 1, k_loop - 1)
                    query_idx_end = p_seg_right_boundary - 1
                    
                    min_prev_sum = st.query_range(query_idx_start, query_idx_end)
                    if min_prev_sum != INF:
                        dp[i][j] = min(dp[i][j], min_prev_sum + nums[j])
            
    result = dp[m-1][n-1]
    return int(result) if result != INF else -1