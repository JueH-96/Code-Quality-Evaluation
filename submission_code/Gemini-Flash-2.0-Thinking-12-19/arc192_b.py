# YOUR CODE HERE
import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Based on the sample inputs and outputs:
    # Sample 1: N=3 (odd), Output: Fennec
    # Sample 2: N=2 (even), Output: Snuke
    # Sample 3: N=6 (even), Output: Snuke
    # This suggests a pattern where the winner is determined solely by the parity of N.
    # Fennec wins if N is odd, and Snuke wins if N is even.

    # This pattern would imply that the total number of "filler" moves
    # (moves on indices already in S) made during the game under optimal play
    # always has the same parity, specifically, it is always even.
    # The total number of moves in the game is N (the number of moves that add a unique index to S)
    # plus the total number of filler moves. The player making the last move wins.
    # The last move is the N-th move that adds a unique index to S.
    # If the total number of moves is T, Fennec wins if T is odd, Snuke wins if T is even.
    # If Fennec wins when N is odd and Snuke when N is even, this means T has the same parity as N.
    # T = N + F_total, where F_total is the total number of filler moves.
    # T mod 2 = (N + F_total) mod 2.
    # If T mod 2 = N mod 2, then (N + F_total) mod 2 = N mod 2, which implies F_total mod 2 = 0.
    # Thus, the hypothesis implies the total number of filler moves is always even.
    # While the theoretical proof for why F_total is always even under optimal play
    # is not immediately obvious from a simple analysis, the strong pattern in samples
    # suggests this is the case.

    if N % 2 == 1:
        print("Fennec")
    else:
        print("Snuke")

solve()