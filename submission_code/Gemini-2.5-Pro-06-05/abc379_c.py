# YOUR CODE HERE
import sys

def solve():
    """
    Reads input, solves the problem, and prints the output.
    """
    try:
        # Use sys.stdin.readline for faster I/O in competitive programming
        input = sys.stdin.readline
        
        # Read problem parameters N (number of cells) and M (number of piles)
        N, M = map(int, input().split())
        
        # Read initial stone positions X and their respective counts A
        X = list(map(int, input().split()))
        A = list(map(int, input().split()))

    except (ValueError, IndexError):
        # Gracefully handle cases with malformed input or end-of-file issues.
        return

    # --- Necessary Condition 1: Total Stones ---
    # To reach a state with exactly one stone in each of the N cells, the total
    # number of stones in the system must be exactly N.
    if sum(A) != N:
        print(-1)
        return

    # Combine positions and counts into pairs and sort them by position.
    # This is crucial for checking the prefix sum condition efficiently.
    pairs = sorted(zip(X, A))

    # --- Necessary and Sufficient Condition 2: Prefix Sums ---
    # Stones can only move forward. This means for any k, the initial number of
    # stones in cells [1, k], let's call it B_k, must be at least k.
    # B_k >= k for all k = 1, ..., N-1.
    # We check this efficiently by only considering critical points, which are
    # right before each pile's position.
    
    cumulative_stones = 0
    for x, a in pairs:
        # `cumulative_stones` holds the total stones in piles at positions < x.
        # This is equivalent to B_{p} where p is the position of the previous pile.
        # The tightest check for the interval before x is at k = x - 1.
        # We need B_{x-1} >= x-1. Since there's no pile at x-1 (X_i are distinct),
        # B_{x-1} is the same as `cumulative_stones`.
        if cumulative_stones < x - 1:
            print(-1)
            return
        
        cumulative_stones += a

    # If the prefix sum conditions are met, a solution is guaranteed to exist.

    # --- Calculation of Minimum Operations ---
    # The total number of operations is the sum of distances each stone moves.
    # Total Operations = Sum over all stones (final_pos - initial_pos)
    #                  = Sum(final_pos) - Sum(initial_pos)
    # This is a change in the total "moment" of the system.
    #
    # Final Moment: One stone at each cell from 1 to N.
    # Sum of positions = 1 + 2 + ... + N = N * (N + 1) / 2.
    #
    # Initial Moment: A_i stones at each position X_i.
    # Sum of initial positions = Sum(A_i * X_i).
    
    initial_moment = 0
    for i in range(M):
        initial_moment += A[i] * X[i]

    final_moment = (N * (N + 1)) // 2
    
    total_ops = final_moment - initial_moment
    
    print(total_ops)

# Run the solution
solve()