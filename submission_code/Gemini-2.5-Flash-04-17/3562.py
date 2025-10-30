from typing import List
import bisect

# Helper to compare states (weight, indices)
# Returns 1 if state1 is better, -1 if state2 is worse, 0 if equal
def compare_states(state1, state2):
    # Invalid state (-1, []) is always worse than a valid state
    if state1[0] == -1 and state2[0] == -1:
        return 0 # Both invalid
    if state1[0] == -1:
        return -1 # state1 is invalid, state2 is better
    if state2[0] == -1:
        return 1 # state2 is invalid, state1 is better

    # Compare by weight
    if state1[0] > state2[0]:
        return 1 # state1 has higher weight, better
    if state1[0] < state2[0]:
        return -1 # state2 has higher weight, better

    # Weights are equal, compare by indices (lexicographically)
    # Note: Indices lists are built by appending, so they are already sorted internally.
    if state1[1] < state2[1]:
        return 1 # state1 has smaller indices list, better
    elif state1[1] > state2[1]:
        return -1 # state2 has smaller indices list, better
    else:
        return 0 # Equal weight and indices

# Segment Tree Node
class SegTreeNode:
    def __init__(self):
        self.best_state = (-1, []) # (weight, indices), initialized to invalid state

# Segment Tree
class SegmentTree:
    def __init__(self, size):
        self.size = size
        # Pad size to the next power of 2
        self.tree_size = 1
        while self.tree_size < size:
            self.tree_size *= 2
        # Tree array: indices 1 to 2*tree_size - 1.
        # Leaves are at indices tree_size to 2*tree_size - 1.
        # Leaf tree_size + i corresponds to data index i (0-based).
        self.tree = [SegTreeNode() for _ in range(2 * self.tree_size)]

    # Update leaf node index 'idx' with 'state' and propagate up
    def update(self, idx, state):
        # Map data index to leaf node index
        leaf_node_idx = self.tree_size + idx
        
        # Merge the new state with the current best state at this leaf
        self.tree[leaf_node_idx].best_state = merge_states(self.tree[leaf_node_idx].best_state, state)

        # Propagate update up the tree
        node = leaf_node_idx // 2
        while node >= 1:
            self.tree[node].best_state = merge_states(
                self.tree[2 * node].best_state,
                self.tree[2 * node + 1].best_state
            )
            node //= 2

    # Query range [l, r] (inclusive) of data indices
    def query(self, l, r):
        return self._query(l, r, 1, 0, self.tree_size - 1)

    def _query(self, l, r, node, start, end):
        # If query range is outside node range
        if r < start or end < l:
            return (-1, []) # Return identity for merge (invalid state)

        # If node range is within query range
        if l <= start and end <= r:
            return self.tree[node].best_state

        # Partial overlap, recurse
        mid = (start + end) // 2
        p1 = self._query(l, r, 2 * node, start, mid)
        p2 = self._query(l, r, 2 * node + 1, mid + 1, end)
        return merge_states(p1, p2)


# Merge function for Segment Tree
def merge_states(state1, state2):
    # Returns the better of state1 and state2
    comp = compare_states(state1, state2)
    if comp >= 0: # state1 is better or equal
        return state1
    else: # state2 is better
        return state2


class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        n = len(intervals)
        # Augment intervals with original index
        augmented_intervals = [[intervals[i][0], intervals[i][1], intervals[i][2], i] for i in range(n)]

        # Sort intervals
        # Sort by start time asc, then end time asc, then weight desc, then index asc
        # This order helps in processing intervals and potentially finding lexicographically smaller index lists earlier
        augmented_intervals.sort(key=lambda x: (x[0], x[1], -x[2], x[3]))

        # Get distinct end times and map them to indices
        # These are the points where intervals can end, which determine the DP states in the Segment Tree
        distinct_end_times = sorted(list(set([itv[1] for itv in augmented_intervals])))
        end_time_map = {time: i for i, time in enumerate(distinct_end_times)}
        D = len(distinct_end_times)

        # If there are no distinct end times, something is wrong with input or constraints, but defensively return empty.
        # With constraints 1 <= l_i <= r_i, D >= 1 if N >= 1.
        if D == 0:
             return []

        # Initialize Segment Trees for k = 1 to 4 intervals
        # segment_trees[k] will store best states using exactly k non-overlapping intervals
        # where the last interval ends at a specific distinct end time index.
        segment_trees = [None] * 5 # Index 0 is unused
        for k in range(1, 5):
            segment_trees[k] = SegmentTree(D)

        # DP using Segment Trees
        # Iterate through sorted intervals
        for l, r, w, idx in augmented_intervals:
            r_idx = end_time_map[r] # Index in distinct_end_times for the current interval's end time

            # Find the index in distinct_end_times for the latest end time strictly less than l
            # bisect_left finds the first index >= l. So index-1 is the last index < l.
            # We are looking for distinct end times that are strictly less than l.
            l_bound_idx = bisect.bisect_left(distinct_end_times, l)
            # The indices in distinct_end_times corresponding to end times < l are [0, l_bound_idx - 1].

            # Iterate through possible number of intervals (k) in the resulting set
            # If we add the current interval I_i, the new set will have k intervals, formed from
            # a previous set of k-1 intervals plus I_i.
            for k in range(1, 5):
                prev_w = -1
                prev_indices = []

                if k == 1:
                    # Base case for forming a 1-interval set:
                    # The "previous" state is a set of 0 intervals, with weight 0 and empty indices [].
                    # This conceptual state can be considered ending at time 0.
                    # Since input constraints are 1 <= l_i, the start time l is always > 0.
                    # Thus, any interval I_i is non-overlapping with this conceptual state ending at time 0.
                    prev_w = 0
                    prev_indices = []

                else:
                    # For k > 1, the previous state comes from a set of k-1 intervals.
                    # We need the best state from T_{k-1} ending strictly before l.
                    # Query the Segment Tree T_{k-1} for the best state in the range of distinct end time indices [0, l_bound_idx - 1].
                    if l_bound_idx > 0: # Ensure the range [0, l_bound_idx - 1] is valid
                         prev_state = segment_trees[k-1].query(0, l_bound_idx - 1)
                         prev_w, prev_indices = prev_state
                    # If l_bound_idx == 0, it means all distinct end times are >= l.
                    # So there is no distinct end time < l, and thus no valid previous state from the tree T_{k-1} ending before l. prev_w remains -1.


                # If a valid previous state is found (weight != -1).
                # Note: for k=1, prev_w is always 0, which is not -1. So this condition holds for k=1 if l > 0.
                if prev_w != -1:
                    new_w = prev_w + w
                    new_indices = prev_indices + [idx] # Append the current interval's original index

                    # Update the Segment Tree T_k at the index corresponding to the current interval's end time (r_idx).
                    # We update T_k with the new state (new_w, new_indices).
                    # The update method handles merging with any existing best state at this index.
                    segment_trees[k].update(r_idx, (new_w, new_indices))


        # After processing all intervals, find the best state from all Segment Trees T_1 to T_4.
        # The best overall state is the one with the maximum weight, breaking ties with the lexicographically smallest indices list.
        best_overall_state = (-1, []) # (weight, indices), initialized to invalid state

        # Query the entire range of distinct end times in each tree T_k to find the best state using exactly k intervals.
        for k in range(1, 5):
            if segment_trees[k].size > 0: # Ensure the tree was initialized
                k_best_state = segment_trees[k].query(0, D - 1)
                # Update overall best state if the state from T_k is better
                if compare_states(k_best_state, best_overall_state) > 0:
                    best_overall_state = k_best_state

        # The result is the indices list from the best overall state.
        # If no intervals could be chosen (e.g., if constraints allowed negative weights or N=0), the weight might be -1.
        # However, with N>=1 and weights>=1, at least one interval can be chosen, resulting in a weight >= 1.
        # The initial (-1, []) ensures any valid state (weight >= 0) is preferred.
        return best_overall_state[1] # Return the indices list