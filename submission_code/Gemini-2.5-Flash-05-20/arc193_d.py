import sys

def solve():
    N = int(sys.stdin.readline())
    A = sys.stdin.readline().strip()
    B = sys.stdin.readline().strip()

    num_a_ones = A.count('1')
    num_b_ones = B.count('1')

    # Condition 1: If target requires more pieces than available, it's impossible.
    if num_a_ones < num_b_ones:
        sys.stdout.write("-1
")
        return

    # current_net_pieces represents the running balance of (pieces from A) - (pieces for B)
    # at or to the left of the current index. This is effectively a prefix sum of (A_k - B_k).
    # This value represents the number of "excess" pieces that need to move to the right
    # (if positive) or the number of "missing" pieces that need to move from the right
    # (if negative) to satisfy the 'B' configuration up to the current point.
    
    # max_net_pieces will store the maximum value of current_net_pieces encountered.
    # min_net_pieces will store the minimum value of current_net_pieces encountered.
    # The minimum number of operations is `max_net_pieces - min_net_pieces`.
    # This range represents the largest "imbalance" of pieces that must be shifted.

    current_net_pieces = 0
    max_net_pieces = 0  # Initialize with 0 to include the prefix sum before any elements
    min_net_pieces = 0  # Initialize with 0 to include the prefix sum before any elements

    for i in range(N):
        if A[i] == '1':
            current_net_pieces += 1
        if B[i] == '1':
            current_net_pieces -= 1
        
        # Update max and min prefix sums
        max_net_pieces = max(max_net_pieces, current_net_pieces)
        min_net_pieces = min(min_net_pieces, current_net_pieces)
    
    sys.stdout.write(f"{max_net_pieces - min_net_pieces}
")


T = int(sys.stdin.readline())
for _ in range(T):
    solve()