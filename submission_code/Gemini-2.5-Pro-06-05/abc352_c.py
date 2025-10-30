import sys

def solve():
    """
    Reads giant data from stdin, calculates the maximum possible stack height based on
    the derived formula, and prints the result to stdout.
    """
    # Read the number of giants. Using sys.stdin.readline is faster for large inputs.
    n = int(sys.stdin.readline())
    
    # The total height for a permutation P = (P_1, ..., P_N) is:
    # H = (A_{P_1} + ... + A_{P_{N-1}}) + B_{P_N}
    # This simplifies to H = (Sum of all A_i) + (B_{P_N} - A_{P_N}).
    # To maximize H, we must find the giant 'i' with the maximum (B_i - A_i)
    # and place them at the top (as P_N). The order of other giants does not matter.

    total_a = 0
    # B_i >= A_i, so the difference is non-negative. Initialize max to 0.
    max_head_shoulder_diff = 0

    # We can calculate the total sum of A_i and the max difference in one pass.
    for _ in range(n):
        a, b = map(int, sys.stdin.readline().split())
        
        # Accumulate the sum of all shoulder heights.
        total_a += a
        
        # Update the maximum head-over-shoulders difference.
        max_head_shoulder_diff = max(max_head_shoulder_diff, b - a)

    # The maximum height is the sum of all A's plus the max difference found.
    # Python's integers handle arbitrary size, so overflow is not an issue.
    result = total_a + max_head_shoulder_diff
    
    print(result)

# Run the solution function.
solve()