# YOUR CODE HERE
import sys

def solve():
    """
    Solves a single test case. Reads the input string and determines if it can
    be transformed into "abc" with at most one swap.
    """
    s = sys.stdin.readline().strip()
    
    # Calculate how many characters are already in their correct positions.
    # Target positions: s[0] == 'a', s[1] == 'b', s[2] == 'c'
    correct_positions_count = 0
    if s[0] == 'a':
        correct_positions_count += 1
    if s[1] == 'b':
        correct_positions_count += 1
    if s[2] == 'c':
        correct_positions_count += 1
        
    # Logic based on the number of correct positions:
    # 3 correct positions: The string is "abc" already (0 swaps needed). Output YES.
    # 1 correct position: The remaining two characters are swapped relative to their target.
    #                     One swap will fix them (1 swap needed). Output YES.
    # 0 correct positions: This is a derangement of 3 elements, which requires 2 swaps.
    #                      This does not meet "at most one operation". Output NO.
    if correct_positions_count == 1 or correct_positions_count == 3:
        sys.stdout.write("YES
")
    else:
        sys.stdout.write("NO
")

# Read the number of test cases
num_test_cases = int(sys.stdin.readline())

# Process each test case
for _ in range(num_test_cases):
    solve()