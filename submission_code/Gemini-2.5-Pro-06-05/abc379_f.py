# YOUR CODE HERE
import sys

def solve():
    """
    Reads input, solves the problem, and prints the output.
    This function encapsulates the entire logic.
    """
    # Use fast I/O
    input = sys.stdin.readline

    # Fenwick Tree (BIT) for 1-based indexing.
    # It supports point updates and prefix sum queries in O(log N) time.
    class FenwickTree:
        def __init__(self, size):
            self.tree = [0] * (size + 1)
            self.size = size

        def add(self, idx, delta):
            """Adds delta to the element at index idx."""
            while idx <= self.size:
                self.tree[idx] += delta
                idx += idx & -idx

        def query(self, idx):
            """Queries the sum of elements from 1 to idx."""
            s = 0
            while idx > 0:
                s += self.tree[idx]
                idx -= idx & -idx
            return s

    # Read problem size and heights
    try:
        N, Q = map(int, input().split())
        H = [0] + list(map(int, input().split())) # 1-based indexing for heights
    except (IOError, ValueError):
        # Handle empty input case for robustness
        return

    # Step 1: Pre-compute `prev_greater` for each building.
    # `prev_greater[k]` stores the index of the nearest building to the west of k
    # that is taller than k. If none, it's 0.
    prev_greater = [0] * (N + 1)
    stack = []
    for k in range(1, N + 1):
        while stack and H[stack[-1]] <= H[k]:
            stack.pop()
        if stack:
            prev_greater[k] = stack[-1]
        stack.append(k)

    # Step 2: Organize queries and data points for offline processing.
    # Group queries by their 'l' value.
    queries_by_l = [[] for _ in range(N + 1)]
    for q_idx in range(Q):
        l, r = map(int, input().split())
        queries_by_l[l].append((r, q_idx))
    
    # Group buildings (k) by their `prev_greater` value.
    points_by_pg = [[] for _ in range(N + 1)]
    for k in range(1, N + 1):
        pg = prev_greater[k]
        points_by_pg[pg].append(k)

    # Step 3: Sweep-line processing.
    bit = FenwickTree(N)
    answers = [0] * Q

    # The sweep-line moves along the 'l' axis from 0 to N.
    for l in range(N + 1):
        # Add all buildings k where `prev_greater[k] == l` into the BIT.
        for k in points_by_pg[l]:
            bit.add(k, 1)
        
        # Process all queries where the `l` constraint is the current sweep-line position.
        if l > 0: # Queries have l >= 1
            for r, q_idx in queries_by_l[l]:
                # We need to count k in the range [r + 1, N].
                # In the BIT, this is `count(k<=N) - count(k<=r)`.
                count = bit.query(N) - bit.query(r)
                answers[q_idx] = count
    
    # Step 4: Print the results for all queries.
    for ans in answers:
        print(ans)

solve()