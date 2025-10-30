import sys
from bisect import bisect_left, bisect_right

# Fenwick tree (Binary Indexed Tree)
class FenwickTree:
    def __init__(self, size):
        # Use 1-based indexing internally for simpler logic
        self.size = size
        self.tree = [0] * (size + 1)

    def update(self, index, value):
        # Update at 0-based rank (map 0-based rank to 1-based index)
        index += 1
        while index <= self.size:
            self.tree[index] += value
            index += index & (-index)

    def query(self, index):
        # Query sum up to 0-based rank (map 0-based rank to 1-based index)
        # query(r) returns sum from rank 0 up to rank r (inclusive)
        index += 1
        prefix_sum = 0
        while index > 0:
            prefix_sum += self.tree[index]
            index -= index & (-index)
        return prefix_sum

    def query_range(self, start_index, end_index):
        # Query sum in 0-based rank range [start_index, end_index]
        # Equivalent to query sum up to end_index minus sum up to start_index - 1
        if start_index > end_index:
            return 0
        # query(end_index) gets sum up to end_index
        # query(start_index - 1) gets sum up to start_index - 1
        # Difference is sum from start_index to end_index
        return self.query(end_index) - self.query(start_index - 1)

def solve():
    N, T = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    X = list(map(int, sys.stdin.readline().split()))

    # Ants move for (T + 0.1) units of time.
    # Collision time t between ant i (X_i, d_i) and ant j (X_j, d_j):
    # X_i + d_i * t = X_j + d_j * t => (X_i - X_j) = (d_j - d_i) * t
    # t = (X_i - X_j) / (d_j - d_i)
    # Ants pass if 0 < t <= T + 0.1 and d_i != d_j.

    # Case 1: Ant i (S_i=1, d_i=1) and Ant j (S_j=0, d_j=-1). d_j - d_i = -2.
    # t = (X_i - X_j) / (-2) = (X_j - X_i) / 2.
    # Pass condition: 0 < (X_j - X_i) / 2 <= T + 0.1
    # 0 < X_j - X_i <= 2 * (T + 0.1)
    # If we consider pair (i, j) with i < j:
    # We need S_i=1, S_j=0, and X_i < X_j, and X_j - X_i <= 2 * (T + 0.1)

    # Case 2: Ant i (S_i=0, d_i=-1) and Ant j (S_j=1, d_j=1). d_j - d_i = 2.
    # t = (X_i - X_j) / 2.
    # Pass condition: 0 < (X_i - X_j) / 2 <= T + 0.1
    # 0 < X_i - X_j <= 2 * (T + 0.1)
    # If we consider pair (i, j) with i < j:
    # We need S_i=0, S_j=1, and X_i > X_j, and X_i - X_j <= 2 * (T + 0.1)
    # This is equivalent to X_j <= X_i - 1 and X_j >= X_i - 2 * (T + 0.1).
    # So X_i - 2 * (T + 0.1) <= X_j < X_i.

    Limit_float = 2.0 * T + 0.2

    # Collect relevant coordinates for compression
    coords_set = set()
    for i in range(N):
        coords_set.add(X[i])
        # Relevant coordinates for boundary checks in range queries
        coords_set.add(X[i] + Limit_float)
        coords_set.add(X[i] - Limit_float)

    # Add coordinates infinitesimally close to the boundaries to handle strict/non-strict inequalities correctly
    # Epsilon approach can be tricky with floating points.
    # A value v is in (A, B] if A < v <= B.
    # A value v is in [A, B) if A <= v < B.
    # bisect_left(coords, v) gives rank of first element >= v.
    # bisect_right(coords, v) gives rank of first element > v.

    # To count elements v in coords such that A < v <= B:
    # Count elements <= B : bisect_right(coords, B) gives index of first element > B. So indices 0 to bisect_right(coords, B) - 1 are <= B. Count is bisect_right(coords, B).
    # Count elements <= A : bisect_right(coords, A) gives index of first element > A. So indices 0 to bisect_right(coords, A) - 1 are <= A. Count is bisect_right(coords, A).
    # Count elements > A and <= B is (count <= B) - (count <= A) = bisect_right(coords, B) - bisect_right(coords, A).
    # The ranks in the Fenwick tree will correspond to indices in the sorted `coords` list.
    # The count of ranks in [rank_start, rank_end] is query(rank_end) - query(rank_start - 1).
    # So, ranks for (A, B] are [bisect_right(coords, A), bisect_right(coords, B) - 1].

    # To count elements v in coords such that A <= v < B:
    # Count elements < B : bisect_left(coords, B) gives index of first element >= B. So indices 0 to bisect_left(coords, B) - 1 are < B. Count is bisect_left(coords, B).
    # Count elements < A : bisect_left(coords, A) gives index of first element >= A. So indices 0 to bisect_left(coords, A) - 1 are < A. Count is bisect_left(coords, A).
    # Count elements >= A and < B is (count < B) - (count < A) = bisect_left(coords, B) - bisect_left(coords, A).
    # The ranks in the Fenwick tree will correspond to indices in the sorted `coords` list.
    # The count of ranks in [rank_start, rank_end] is query(rank_end) - query(rank_start - 1).
    # So, ranks for [A, B) are [bisect_left(coords, A), bisect_left(coords, B) - 1].

    coords = sorted(list(coords_set))
    M = len(coords)

    # Map coordinate value to its compressed rank (0-based index in sorted coords)
    # map_val_to_rank = lambda val: bisect_left(coords, val) # This works if val is in coords

    ft_R = FenwickTree(M) # Stores ranks of X_j for right-movers j with original index > k
    ft_L = FenwickTree(M) # Stores ranks of X_j for left-movers j with original index > k

    total_count = 0

    # Iterate through ants in reverse order of original index k = 0..N-1
    # When processing ant k, the FTs contain information about ants j with original index > k
    for k in range(N - 1, -1, -1):
        x_k = X[k]
        s_k = S[k]
        # Get the rank of the current ant's position in the compressed coordinate list
        rank_x_k = bisect_left(coords, x_k)

        if s_k == '1': # Current ant k moves right (S_k=1)
            # Looking for pairs (k, j) with k < j, S[j] == '0', X[j] in (X[k], X[k] + Limit_float].
            # k < j means ant j has original index > k. These are left-movers stored in ft_L.
            # Query ft_L for count of ants j (with original index > k) whose positions X[j]
            # fall within the range (x_k, x_k + Limit_float].
            
            # Range (A, B] -> ranks [bisect_right(coords, A), bisect_right(coords, B) - 1].
            
            rank_lower_exc_coord = bisect_right(coords, x_k)
            rank_upper_inc_coord = bisect_right(coords, x_k + Limit_float) - 1

            # Query ft_L for count of ants with ranks in [rank_lower_exc_coord, rank_upper_inc_coord]
            total_count += ft_L.query_range(rank_lower_exc_coord, rank_upper_inc_coord)

            # Add current ant k (right-mover) to ft_R for future queries (by ants with index < k).
            # Store its position rank.
            ft_R.update(rank_x_k, 1)

        else: # Current ant k moves left (s_k == '0')
            # Current ant k moves left. Looking for pairs (k, j) with k < j, S[j] == '1', X[j] in [X[k] - Limit_float, X[k]).
            # k < j means ant j has original index > k. These are right-movers stored in ft_R.
            # Query ft_R for count of ants j (with original index > k) whose positions X[j]
            # fall within the range [x_k - Limit_float, x_k).

            # Range [A, B) -> ranks [bisect_left(coords, A), bisect_left(coords, B) - 1].

            rank_lower_inc_coord = bisect_left(coords, x_k - Limit_float)
            rank_upper_exc_coord = bisect_left(coords, x_k) - 1

            # Query ft_R for count of ants with ranks in [rank_lower_inc_coord, rank_upper_exc_coord]
            total_count += ft_R.query_range(rank_lower_inc_coord, rank_upper_exc_coord)

            # Add current ant k (left-mover) to ft_L for future queries (by ants with index < k).
            # Store its position rank.
            ft_L.update(rank_x_k, 1)

    print(total_count)

solve()