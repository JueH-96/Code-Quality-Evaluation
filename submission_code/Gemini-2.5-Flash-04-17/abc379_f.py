import sys

# Fenwick Tree (BIT) implementation for 0-indexed values
class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (size + 1)

    def update(self, index, value):
        # index is 0-based value index, convert to 1-based tree index
        index += 1
        while index <= self.size:
            self.tree[index] += value
            index += index & (-index)

    def query(self, index):
        # index is 0-based value index, get sum for values <= index
        # query(v) gives sum for values from 0 up to v inclusive
        index += 1
        sum_val = 0
        while index > 0:
            sum_val += self.tree[index]
            index -= index & (-index)
        return sum_val

def solve():
    # Read N and Q
    N, Q = map(int, sys.stdin.readline().split())
    # Read heights H, 0-indexed list
    H = list(map(int, sys.stdin.readline().split()))

    # Compute prev_greater[j]: largest index k < j such that H[k] > H[j]
    # Initialize with -1 (no such element)
    prev_greater = [-1] * N
    stack = [] # Stores indices k in increasing order, H[k] in decreasing order
    for j in range(N):
        # Pop elements from stack while H[stack[-1]] < H[j]
        while stack and H[stack[-1]] < H[j]:
            stack.pop()
        # If stack is not empty, top is the largest index k < j with H[k] > H[j]
        if stack:
            prev_greater[j] = stack[-1]
        # Push current index j onto the stack
        stack.append(j)

    # Transform prev_greater values to be non-negative for BIT
    # Map [-1, N-2] to [0, N-1] by adding 1
    # pg_prime[j] = prev_greater[j] + 1
    pg_prime = [pg + 1 for pg in prev_greater]

    # Prepare sweep-line events
    events = []
    # Add point events (j, pg_prime[j]) for each building j
    # Event type 0: point
    # Format: (x_coord, y_coord, type, sign, original_query_idx)
    for j in range(N):
        events.append((j, pg_prime[j], 0, 1, -1)) # type 0 for point, sign 1 for adding

    # Add query events for each query (l_i, r_i)
    # Convert l_i, r_i from 1-based to 0-based
    # Query (L, R) asks for count of j in [R+1, N-1] s.t. prev_greater[j] <= L
    # This is equivalent to count of j in [R+1, N-1] s.t. pg_prime[j] <= L+1
    # This count = (count of points (j', pg'[j']) with j' <= N-1 and pg'[j'] <= L+1)
    #           - (count of points (j', pg'[j']) with j' <= R and pg'[j'] <= L+1)
    # Using F(x, y) = count of points (j', pg'[j']) with j' <= x and pg'[j'] <= y
    # We need F(N-1, L+1) - F(R, L+1)
    for i in range(Q):
        l, r = map(int, sys.stdin.readline().split())
        L = l - 1 # 0-indexed l
        R = r - 1 # 0-indexed r

        y_coord = L + 1 # Upper bound for pg' values, 0-indexed [0, N-1]

        # Event at x = R, subtract contribution for F(R, L+1)
        # Query type 1, sign -1
        events.append((R, y_coord, 1, -1, i))

        # Event at x = N-1, add contribution for F(N-1, L+1)
        # Query type 1, sign 1
        events.append((N - 1, y_coord, 1, 1, i))

    # Sort events: by x-coordinate ascending, then by type ascending (points before queries)
    events.sort()

    # Sweep-line with Fenwick tree
    # The BIT operates on the y-coordinates (pg_prime values), which are in [0, N-1].
    # The size of the BIT should be N to accommodate indices 0 to N-1.
    ft = FenwickTree(N)
    ans = [0] * Q

    for x_coord, y_coord, event_type, sign, query_idx in events:
        if event_type == 0: # Point event
            # y_coord is pg_prime[j], which is in range [0, N-1]
            ft.update(y_coord, 1)
        else: # Query event
            # y_coord is L+1. We need sum of counts for pg' values <= L+1.
            # ft.query(y_coord) correctly calculates the sum for values from 0 up to y_coord inclusive.
            ans[query_idx] += sign * ft.query(y_coord)

    # Print results
    for res in ans:
        print(res)

solve()