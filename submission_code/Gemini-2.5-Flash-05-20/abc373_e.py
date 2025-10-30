import sys
import bisect

# Helper class for Segment Tree node
class SegTreeNode:
    def __init__(self):
        self.count = 0
        self.sum_val = 0

# Segment Tree implementation
class SegmentTree:
    def __init__(self, size, unique_vals):
        self.size = size
        self.unique_vals = unique_vals
        # tree is 1-indexed for convenience in typical segment tree operations
        self.tree = [SegTreeNode() for _ in range(4 * size)]

    # Update operation for adding/removing a candidate's votes
    # delta_count: +1 for add, -1 for remove
    def _update(self, node_idx, node_start, node_end, update_idx, delta_count, value):
        if node_start == node_end:
            self.tree[node_idx].count += delta_count
            self.tree[node_idx].sum_val += delta_count * value
            return

        mid = (node_start + node_end) // 2
        if node_start <= update_idx <= mid:
            self._update(2 * node_idx, node_start, mid, update_idx, delta_count, value)
        else:
            self._update(2 * node_idx + 1, mid + 1, node_end, update_idx, delta_count, value)
        
        # Aggregate results from children
        self.tree[node_idx].count = self.tree[2 * node_idx].count + self.tree[2 * node_idx + 1].count
        self.tree[node_idx].sum_val = self.tree[2 * node_idx].sum_val + self.tree[2 * node_idx + 1].sum_val

    def update(self, rank, value, delta_count):
        self._update(1, 0, self.size - 1, rank, delta_count, value)

    # Range query for count and sum of votes
    def _query(self, node_idx, node_start, node_end, query_start, query_end):
        # If query range is completely outside node range
        if query_start > node_end or query_end < node_start:
            return 0, 0
        # If query range completely covers node range
        if query_start <= node_start and node_end <= query_end:
            return self.tree[node_idx].count, self.tree[node_idx].sum_val

        # Partial overlap, recurse
        mid = (node_start + node_end) // 2
        count_left, sum_left = self._query(2 * node_idx, node_start, mid, query_start, query_end)
        count_right, sum_right = self._query(2 * node_idx + 1, mid + 1, node_end, query_start, query_end)
        return count_left + count_right, sum_left + sum_right

    def query(self, query_start_rank, query_end_rank):
        if query_start_rank > query_end_rank: # Empty range
            return 0, 0
        return self._query(1, 0, self.size - 1, query_start_rank, query_end_rank)

    # Custom function to find max 'k' candidates with A_j <= unique_vals[actual_max_rank]
    # that can be pushed above target_plus_1 (score_i + 1), given a budget.
    # Returns (k, total_cost_to_push_them)
    def _find_max_pushable(self, node_idx, node_start_rank, node_end_rank, budget, target_plus_1, actual_max_rank):
        # Base cases:
        # 1. Node's range is completely above the `actual_max_rank` (meaning A_j > score_i)
        # 2. No budget left
        # 3. No candidates in this node's range
        if node_start_rank > actual_max_rank or budget == 0 or self.tree[node_idx].count == 0:
            return 0, 0

        # If we can afford to push all candidates in this node's range
        # Total cost for all candidates in this node: sum(target_plus_1 - A_j)
        # = (count of candidates in node) * target_plus_1 - (sum of A_j in node)
        total_cost_node = self.tree[node_idx].count * target_plus_1 - self.tree[node_idx].sum_val
        if total_cost_node <= budget:
            return self.tree[node_idx].count, total_cost_node
        
        # If it's a leaf node and we couldn't afford it (checked by previous if), then we can't push any
        if node_start_rank == node_end_rank:
            return 0, 0 

        mid = (node_start_rank + node_end_rank) // 2
        
        # Prioritize recursive call to the right child (higher A_j values, thus lower cost to push)
        k_right, cost_right = 0, 0
        # Check if right child's range is relevant (overlaps with [node_start_rank, actual_max_rank])
        if mid + 1 <= actual_max_rank: 
            k_right, cost_right = self._find_max_pushable(2 * node_idx + 1, mid + 1, node_end_rank, budget, target_plus_1, actual_max_rank)
        
        # Recurse on left child with remaining budget
        remaining_budget = budget - cost_right
        k_left, cost_left = 0, 0
        # Check if left child's range is relevant and if budget allows
        if node_start_rank <= actual_max_rank and remaining_budget > 0:
            k_left, cost_left = self._find_max_pushable(2 * node_idx, node_start_rank, mid, remaining_budget, target_plus_1, actual_max_rank)
        
        return k_right + k_left, cost_right + cost_left

    def find_max_pushable(self, actual_max_rank, budget, target_plus_1):
        # Start recursion from the root of the segment tree
        return self._find_max_pushable(1, 0, self.size - 1, budget, target_plus_1, actual_max_rank)


def solve():
    N, M, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    S = sum(A)
    R = K - S # Remaining total votes to be distributed

    # Coordinate compression: Map original A_j values to ranks (0 to num_unique-1)
    unique_A_values_set = set(A)
    unique_A_values = sorted(list(unique_A_values_set))
    rank_map = {val: i for i, val in enumerate(unique_A_values)}
    num_unique = len(unique_A_values)

    # Initialize segment tree with all A_j values from all candidates
    seg_tree = SegmentTree(num_unique, unique_A_values)
    for val in A:
        seg_tree.update(rank_map[val], val, 1)

    results = [0] * N

    # Process each candidate to find their minimum X
    for cand_idx in range(N):
        # Get the current candidate's vote count and its rank
        cand_A_val = A[cand_idx]
        cand_rank = rank_map[cand_A_val]
        
        # Temporarily remove candidate A[cand_idx] from the segment tree.
        # This makes segment tree queries reflect the state of *other* candidates.
        seg_tree.update(cand_rank, cand_A_val, -1)

        # check(X) function to determine if X additional votes guarantee victory for cand_idx
        def check(X):
            score_i = cand_A_val + X
            remaining_budget_for_others = R - X
            
            # If X requires more votes than available in total, it's impossible
            if remaining_budget_for_others < 0:
                return False

            # Find the rank threshold: A_j values with ranks >= idx_cutoff_rank are > score_i
            # A_j values with ranks < idx_cutoff_rank are <= score_i
            idx_cutoff_rank = bisect.bisect_right(unique_A_values, score_i)
            
            # 1. Count candidates j != cand_idx who already have A_j > score_i
            # These are candidates in the segment tree with ranks from idx_cutoff_rank to num_unique-1
            count_already_above, _ = seg_tree.query(idx_cutoff_rank, num_unique - 1)
            
            total_beaten_count = count_already_above

            # 2. For candidates j != cand_idx whose A_j <= score_i:
            # We want to use `remaining_budget_for_others` to push as many of them as possible
            # above `score_i`. We prioritize those with higher `A_j` (lower cost to push).
            # The `find_max_pushable` function does this efficiently.
            # `actual_max_rank` for pushing means we consider candidates with ranks from 0 up to `idx_cutoff_rank - 1`.
            max_pushable_k, _ = seg_tree.find_max_pushable(idx_cutoff_rank - 1, remaining_budget_for_others, score_i + 1)
            
            total_beaten_count += max_pushable_k

            # Candidate i wins if the total number of candidates beating them is less than M
            return total_beaten_count < M

        # Binary search for the minimum X for current candidate
        low = 0
        high = R + 1 # Search range for X. X can be 0 up to R. R+1 ensures R is included.
        ans = -1

        while low <= high:
            mid = low + (high - low) // 2
            if check(mid):
                ans = mid
                high = mid - 1 # Try a smaller X
            else:
                low = mid + 1 # Need a larger X
        
        results[cand_idx] = ans
        
        # Restore candidate A[cand_idx] to the segment tree before processing the next candidate.
        seg_tree.update(cand_rank, cand_A_val, 1)

    # Print results
    sys.stdout.write(" ".join(map(str, results)) + "
")

solve()