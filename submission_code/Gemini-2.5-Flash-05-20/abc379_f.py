import sys

# Fenwick Tree (BIT) implementation for range queries
# Uses 0-based indexing for external calls, converts to 1-based internally.
class FenwickTree:
    def __init__(self, size):
        # The tree array uses 1-based indexing internally.
        # `size` corresponds to the maximum 0-based index + 1.
        self.tree = [0] * (size + 1)
        self.size = size

    def update(self, idx, delta):
        # Updates the value at 0-based index `idx` by `delta`.
        idx += 1  # Convert to 1-based index for internal tree operations
        while idx <= self.size:
            self.tree[idx] += delta
            idx += (idx & -idx)

    def query_prefix(self, idx):
        # Gets the sum of values from 0-based index 0 to `idx` (inclusive).
        idx += 1  # Convert to 1-based index for internal tree operations
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= (idx & -idx)
        return res

    def query_range(self, start_idx, end_idx):
        # Gets the sum of values for 0-based range [start_idx, end_idx] (inclusive).
        if start_idx > end_idx:
            return 0
        return self.query_prefix(end_idx) - self.query_prefix(start_idx - 1)

def solve():
    # Read N and Q from standard input
    N, Q = map(int, sys.stdin.readline().split())

    # Read building heights H (0-indexed for convenience)
    H = list(map(int, sys.stdin.readline().split()))

    # Calculate R_arr: R_arr[k] is the largest index j < k such that H[j] > H[k].
    # If no such j exists, R_arr[k] = -1.
    # This is done using a monotonic stack (decreasing sequence of heights).
    R_arr = [-1] * N
    stack = [] # Stores 0-based indices of buildings
    for k in range(N):
        # Pop elements from stack whose heights are less than H[k].
        # These elements are "hidden" by H[k] from their right.
        while stack and H[stack[-1]] < H[k]:
            stack.pop()
        
        # If the stack is not empty, the top element is the index of the previous greater height.
        if stack:
            R_arr[k] = stack[-1]
        else:
            # If stack is empty, H[k] is the tallest among H[0...k], so no previous greater element.
            R_arr[k] = -1
        
        # Push current index k onto the stack.
        stack.append(k)

    # Prepare events for offline processing.
    # An event is either a 'point' or a 'query'.
    # 'point' event: ('point', threshold_val=R_arr[x_idx], data_val=x_idx)
    # 'query' event: ('query', threshold_val=l_0idx, data_val=r_0idx, original_query_idx)
    # The 'threshold_val' is the value by which we will sort events.
    
    events = []
    # Add all N points (R_arr[x_idx], x_idx)
    for x_idx in range(N):
        events.append(('point', R_arr[x_idx], x_idx))

    # Read Q queries and add them to the events list
    # Queries are (l, r) (1-indexed). Convert to 0-indexed (l_0idx, r_0idx).
    # A query needs to count x_0 such that:
    #   1. r_0idx < x_0 < N (i.e., x_0 in range [r_0idx + 1, N-1])
    #   2. R_arr[x_0] <= l_0idx
    # The l_0idx serves as the threshold for sorting.
    
    # This array will store answers for each query in the original order.
    answers = [0] * Q

    for q_idx in range(Q):
        l, r = map(int, sys.stdin.readline().split())
        l_0idx = l - 1
        r_0idx = r - 1
        events.append(('query', l_0idx, r_0idx, q_idx))

    # Sort events.
    # Primary sort key: the 'threshold_val' (R_arr[x_idx] for points, l_0idx for queries).
    # Secondary sort key: event type. 'point' events must be processed before 'query' events
    # if their threshold_val is the same. This ensures that points that satisfy the current
    # threshold are added to the Fenwick Tree before any query depending on that threshold.
    # (event[0] == 'query') evaluates to False for 'point' and True for 'query',
    # so 'point' events will come before 'query' events if thresholds are equal.
    events.sort(key=lambda event: (event[1], event[0] == 'query'))

    # Initialize the Fenwick Tree
    ft = FenwickTree(N)

    # Process events in sorted order
    for event_type, threshold_val, *rest in events:
        if event_type == 'point':
            x_idx = rest[0]
            # This point's R_arr[x_idx] value is `threshold_val`.
            # When processing, it means all queries with l_0idx >= threshold_val
            # will consider this point. So, add it to the BIT at its `x_idx`.
            ft.update(x_idx, 1)
        elif event_type == 'query':
            l_0idx = threshold_val # This is the l_0idx for the current query.
            r_0idx, q_idx = rest[0], rest[1]
            
            # The query asks for points x_0 such that:
            # 1. R_arr[x_0] <= l_0idx (handled by the sorting and processing order)
            # 2. x_0 is in the range [r_0idx + 1, N-1] (inclusive, 0-indexed).
            answers[q_idx] = ft.query_range(r_0idx + 1, N - 1)

    # Print all answers to standard output
    for res in answers:
        sys.stdout.write(str(res) + "
")

# Call the solve function to run the program
solve()