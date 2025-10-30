import sys

# Define Segment Tree Node structure
class SegTreeNode:
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

class SegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [None] * (4 * self.n)
        self.arr = arr # Reference to the underlying array (S_final_int)
        self._build(0, 0, self.n - 1)

    def _build(self, node_idx, start, end):
        if start == end:
            val = self.arr[start]
            self.tree[node_idx] = SegTreeNode(val, val)
        else:
            mid = (start + end) // 2
            self._build(2 * node_idx + 1, start, mid)
            self._build(2 * node_idx + 2, mid + 1, end)
            left_child = self.tree[2 * node_idx + 1]
            right_child = self.tree[2 * node_idx + 2]
            
            # Combine min/max values from children
            self.tree[node_idx] = SegTreeNode(
                min(left_child.min_val, right_child.min_val),
                max(left_child.max_val, right_child.max_val)
            )

    def update(self, idx, val):
        self._update(0, 0, self.n - 1, idx, val)

    def _update(self, node_idx, start, end, idx, val):
        if start == end:
            self.arr[idx] = val # Update actual array value
            self.tree[node_idx] = SegTreeNode(val, val)
        else:
            mid = (start + end) // 2
            if start <= idx <= mid:
                self._update(2 * node_idx + 1, start, mid, idx, val)
            else:
                self._update(2 * node_idx + 2, mid + 1, end, idx, val)
            left_child = self.tree[2 * node_idx + 1]
            right_child = self.tree[2 * node_idx + 2]
            self.tree[node_idx] = SegTreeNode(
                min(left_child.min_val, right_child.min_val),
                max(left_child.max_val, right_child.max_val)
            )

    # Query for leftmost index `i` such that `arr[i] < threshold` in range [query_start, query_end]
    # Returns -1 if no such index exists.
    def query_leftmost_less_than(self, node_idx, start, end, query_start, query_end, threshold):
        # Current segment is outside query range
        if query_start > end or query_end < start:
            return -1
        
        # If current segment values are all >= threshold, no improvement possible here
        if self.tree[node_idx].max_val < threshold:
            return -1

        # If it's a leaf node and value is < threshold, return its index
        if start == end:
            return start if self.arr[start] < threshold else -1

        mid = (start + end) // 2
        
        # Search left child first (for leftmost)
        res = self.query_leftmost_less_than(2 * node_idx + 1, start, mid, query_start, query_end, threshold)
        if res != -1:
            return res
        
        # If not found in left, search right child
        return self.query_leftmost_less_than(2 * node_idx + 2, mid + 1, end, query_start, query_end, threshold)

    # Query for rightmost index `i` such that `arr[i] < threshold` in range [query_start, query_end]
    # Returns -1 if no such index exists.
    def query_rightmost_less_than(self, node_idx, start, end, query_start, query_end, threshold):
        # Current segment is outside query range
        if query_start > end or query_end < start:
            return -1
            
        # If current segment values are all >= threshold, no improvement possible here
        if self.tree[node_idx].max_val < threshold:
            return -1

        # If it's a leaf node and value is < threshold, return its index
        if start == end:
            return start if self.arr[start] < threshold else -1

        mid = (start + end) // 2
        
        # Search right child first (for rightmost)
        res = self.query_rightmost_less_than(2 * node_idx + 2, mid + 1, end, query_start, query_end, threshold)
        if res != -1:
            return res
        
        # If not found in right, search left child
        return self.query_rightmost_less_than(2 * node_idx + 1, start, mid, query_start, query_end, threshold)

# Main logic
def solve():
    N, M = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    # Convert strings to lists of integers for easier manipulation
    S_final_int = [int(c) for c in S]
    T_char_vals = [int(c) for c in T]

    # Precompute suffix maximums for T_char_vals
    t_suffix_max = [0] * M # Stores max value from index k to M-1
    if M > 0:
        t_suffix_max[M - 1] = T_char_vals[M - 1]
        for k in range(M - 2, -1, -1):
            t_suffix_max[k] = max(T_char_vals[k], t_suffix_max[k+1])

    # Initialize Segment Tree with the initial S_final_int values
    seg_tree = SegmentTree(S_final_int) # S_final_int is passed by reference and modified by seg_tree.update

    # Perform M operations
    for k in range(M):
        char_T = T_char_vals[k]
        
        # max_future_T: largest digit in T[k+1 ... M-1]
        # If k is M-1, there are no future chars, so consider max_future_T as 0.
        # This makes the condition char_T >= max_future_T true for the last character,
        # correctly prioritizing i_L in that final step.
        max_future_T = 0
        if k + 1 < M:
            max_future_T = t_suffix_max[k+1]

        # Find leftmost index i_L such that S_final_int[i_L] < char_T
        i_L = seg_tree.query_leftmost_less_than(0, 0, N - 1, 0, N - 1, char_T)
        
        # Find rightmost index i_R such that S_final_int[i_R] < char_T
        i_R = seg_tree.query_rightmost_less_than(0, 0, N - 1, 0, N - 1, char_T)

        target_idx = -1
        if i_L == -1:
            # Case 1: No position in S_final_int can be improved by char_T.
            # All S_final_int[i] are >= char_T. To minimize value degradation,
            # replace the rightmost character (least significant position).
            target_idx = N - 1 if N > 0 else -1 # Handle N=0 case defensively, though N>=1 by constraints.
        else:
            # Case 2: An improvement is possible.
            # Decide between i_L and i_R based on the comparison with max_future_T.
            if char_T >= max_future_T:
                # If char_T is as good as or better than any future digit for a significant position,
                # use it to improve the leftmost possible position.
                target_idx = i_L
            else:
                # If a larger digit might come in the future that could improve S[i_L],
                # then "save" S[i_L] for that potential future digit.
                # Use char_T to improve the rightmost possible position.
                target_idx = i_R 
        
        # Perform the update if a valid target index was found
        if target_idx != -1:
            seg_tree.update(target_idx, char_T)

    # Convert the final list of integers back to a string and print
    sys.stdout.write("".join(map(str, S_final_int)) + "
")

solve()