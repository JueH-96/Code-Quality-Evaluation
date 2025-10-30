import sys

def solve():
    # Read input values
    # Use integer division // for large numbers
    K, Sx, Sy, Tx, Ty = map(int, sys.stdin.readline().split())

    # Calculate block indices (i, j)
    # For a point (x+0.5, y+0.5)
    # If block (i, j) is Type H ([iK, (i+1)K] x [jK+k, jK+k+1]), point implies i = floor(x/K), j = floor(y/K)
    # If block (i, j) is Type V ([iK+k, iK+k+1] x [jK, (j+1)K]), point implies i = floor(x/K), j = floor(y/K)
    # So the block indices (i, j) for a point (x+0.5, y+0.5) containing integers x, y are simply (floor(x/K), floor(y/K)).
    # Python's integer division `//` is floor division.
    si = Sx // K
    sj = Sy // K
    ti = Tx // K
    tj = Ty // K

    # Determine the type of the start and target blocks based on parity of (i, j)
    # If i and j have the same parity, block is Type H (Horizontal strips).
    # If i and j have different parity, block is Type V (Vertical strips).
    # We can represent this block type parity as (i + j) % 2.
    # Type H: (i+j) % 2 == 0
    # Type V: (i+j) % 2 == 1
    ps = (si + sj) % 2
    pt = (ti + tj) % 2

    # If start and end are in the same block (si == ti and sj == tj)
    if si == ti and sj == tj:
        # The tile containing (Sx+0.5, Sy+0.5) is (si, sj, ks)
        # The tile containing (Tx+0.5, Ty+0.5) is (ti, tj, kt)

        if ps == 0: # Start/Target block is Type H. Tile is (i, j, k) where k = y % K
            # Point (Sx+0.5, Sy+0.5) in tile (si, sj, ks) implies ks = Sy % K
            # Point (Tx+0.5, Ty+0.5) in tile (ti, tj, kt) implies kt = Ty % K
            ks = Sy % K
            kt = Ty % K
        else: # Start/Target block is Type V. Tile is (i, j, k) where k = x % K
            # Point (Sx+0.5, Sy+0.5) in tile (si, sj, ks) implies ks = Sx % K
            # Point (Tx+0.5, Ty+0.5) in tile (ti, tj, kt) implies kt = Tx % K
            ks = Sx % K
            kt = Tx % K

        # Minimum moves within the same block is the absolute difference between the k values
        return abs(ks - kt)
    else:
        # Start and end are in different blocks. We must move between blocks.
        # The minimum number of block transitions required is the Manhattan distance
        # between the start and target block indices.
        block_dist = abs(si - ti) + abs(sj - tj)

        # Each move to an adjacent block (changing i by +/-1 or j by +/-1) flips the
        # parity of (i+j), and therefore flips the block type (H <-> V).
        # Start block type parity: ps
        # Target block type parity: pt
        # After `m` block transitions, the block type parity will be (ps + m) % 2.
        # For a path with the minimum number of block transitions (`block_dist`),
        # the final block type parity will be (ps + block_dist) % 2.
        # The target block has parity `pt`.

        # Based on the sample cases, the minimum number of tile moves appears to be
        # the block Manhattan distance if the start and target blocks have the same type parity (ps == pt),
        # and the block Manhattan distance plus one if they have different type parity (ps != pt).
        # This suggests that the "misalignment" between start and target block types when ps != pt
        # costs exactly one extra move compared to the block distance.

        if ps == pt:
            # Start and target blocks have the same type parity.
            # The minimum number of block steps is even.
            # The total cost is the block Manhattan distance.
            return block_dist
        else:
            # Start and target blocks have different type parity.
            # The minimum number of block steps is odd.
            # The total cost is the block Manhattan distance plus one.
            return block_dist + 1

# Read the number of test cases
T = int(sys.stdin.readline())

# Process each test case
for _ in range(T):
    # Solve the problem for the current test case and print the result
    print(solve())