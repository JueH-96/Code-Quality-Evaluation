import sys

def solve():
    # Read n, k for this test case
    n, k = map(int, sys.stdin.readline().split())
    # Read string s for this test case
    s = sys.stdin.readline().strip()

    operations_count = 0
    current_index = 0
    while current_index < n:
        if s[current_index] == 'B':
            # Found a black cell. We must perform an operation.
            operations_count += 1
            # This operation covers s[current_index] through s[min(current_index + k - 1, n - 1)].
            # All these cells become white.
            # So, we can skip checking these k cells (or fewer if near the end) 
            # and continue scanning from current_index + k.
            current_index += k
        else:
            # This cell is white. Move to the next cell.
            current_index += 1
    
    # Print the result for this test case
    sys.stdout.write(str(operations_count) + "
")

# Read the number of test cases
num_test_cases = int(sys.stdin.readline())

# Process each test case
for _ in range(num_test_cases):
    solve()