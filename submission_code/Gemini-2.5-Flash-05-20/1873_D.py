import sys

def solve():
    n, k = map(int, sys.stdin.readline().split())
    s = sys.stdin.readline().strip()

    operations = 0
    i = 0
    while i < n:
        if s[i] == 'B':
            # Found a black cell. This requires an operation.
            operations += 1
            # Apply an operation starting at index i.
            # This operation covers cells [i, i + k - 1].
            # All cells in this range become white.
            # Therefore, we can skip directly to checking the cell
            # immediately after this covered segment.
            i += k
        else:
            # If s[i] is already white, no operation is needed for this cell.
            # Just move to the next cell.
            i += 1
    
    sys.stdout.write(str(operations) + '
')

# Read the number of test cases
t = int(sys.stdin.readline())

# Process each test case
for _ in range(t):
    solve()