import sys

# Read input
N = int(sys.stdin.readline())
A = list(map(int, sys.stdin.readline().split()))

# If N=1, the set S becomes {1} after the first move.
# Fennec makes the first move, so Fennec wins immediately regardless of A_1 (as A_1 >= 1).
if N == 1:
    print("Fennec")
else:
    # For N > 1, analyze the moves.
    # A move consists of choosing index i such that A_i >= 1, decrementing A_i.
    # If i is not in S, it's added to S. The game ends when S = {1, ..., N}.
    # The player who makes the move that completes S wins.
    # This means the winner is determined by the parity of the total number of moves made.
    # Fennec wins if the total number of moves is odd. Snuke wins if it's even.

    # An index i is added to S on the first move that chooses i.
    # There are N such "first moves" that must happen eventually for S to become {1, ..., N}.
    # For an index i with A_i = 1, the first move sets A_i to 0, and no more moves are possible on i.
    # This first move is a "task completion" move and provides no "free moves" thereafter on this index.
    # For an index i with A_i > 1, the first move adds i to S, and leaves A_i - 1 moves remaining on index i.
    # These A_i - 1 subsequent moves (choosing i when i is already in S) are "free moves" in the sense
    # that they do not add a new index to S. They just consume the value A_i.

    # The total number of moves T is the sum of how many times each index i was chosen (let this be c_i)
    # until S becomes {1, ..., N}. T = sum(c_i).
    # For S to be {1, ..., N}, each index must have been chosen at least once, so c_i >= 1 for all i.
    # The total number of moves T = sum(c_i) = sum(1 + (c_i - 1)) = N + sum(c_i - 1).
    # sum(c_i - 1) is the total number of moves made that were *not* the very first move on an index.
    # If A_i = 1, c_i must be 1 (since A_i becomes 0 after the first move). So c_i - 1 = 0 for indices with initial A_i = 1.
    # If A_i > 1, c_i can be anywhere from 1 up to initial A_i. So c_i - 1 can be from 0 up to initial A_i - 1.
    # The total number of "free moves" made is sum(c_i - 1) for indices where initial A_i > 1.
    # The maximum possible number of such free moves is sum(initial A_i - 1) for indices where initial A_i > 1.
    # Let M_prime = sum(A_i - 1) for all i where A_i > 1. This is the total pool of moves available
    # beyond the first move on indices that support such additional moves.

    # The key insight from sample cases suggests that for N > 1, the winner is determined by
    # the parity of M_prime. If the total potential number of "free moves" on indices with A_i > 1
    # is odd, Fennec wins. If it's even, Snuke wins.

    sum_ai_minus_1_greater_than_1 = 0
    for a in A:
        if a > 1:
            sum_ai_minus_1_greater_than_1 += (a - 1)

    # The winner is determined by the parity of this sum.
    if sum_ai_minus_1_greater_than_1 % 2 == 1:
        print("Fennec")
    else:
        print("Snuke")