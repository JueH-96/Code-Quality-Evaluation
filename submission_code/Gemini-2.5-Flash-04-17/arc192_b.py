import sys

# The solve() function contains the core logic.
def solve():
    # Read N
    N = int(sys.stdin.readline())
    # Read A_1, ..., A_N
    A = list(map(int, sys.stdin.readline().split()))

    # In this game, players need to make N "useful" moves (picking each index for the first time)
    # to add all indices to set S and end the game. The player making the N-th useful move wins.
    # Players can also make "wasted" moves (picking an index already in S).
    # A wasted move on index i is only possible if i is in S and A_i >= 1.
    # The first pick of index i uses 1 from A_i and adds i to S.
    # Subsequent picks of index i use 1 from A_i but do not change S.
    # The total number of moves is M = N + T, where T is the total number of wasted moves made.
    # Fennec wins if M is odd, Snuke wins if M is even.

    # Consider the parity of the number of potential wasted moves on each index after its first pick.
    # For index i, the number of potential wasted moves is A_i - 1 (if A_i >= 1, else 0).
    # If A_i = 1, potential wasted moves = 0 (even). A_i is odd.
    # If A_i > 1 and A_i is odd, potential wasted moves = A_i - 1 (even). A_i is odd.
    # If A_i > 1 and A_i is even, potential wasted moves = A_i - 1 (odd). A_i is even.

    # Only indices with an initially even A_i value (which must be >= 2 since A_i >= 1)
    # provide an odd number of potential wasted moves after being picked once.
    # These indices allow players to potentially flip the parity of the total number of wasted moves made using that index.
    # The total number of such "parity-flipping" sources (indices with even A_i) seems to determine the winner.
    # If there is an odd number of such sources, the first player (Fennec) can use this to their advantage to control the total moves' parity.
    # If there is an even number of such sources, the second player (Snuke) can counter Fennec's attempts, negating the first player advantage from these sources.

    # Count the number of A_i values that are even.
    even_a_count = 0
    for val in A:
        if val % 2 == 0:
            even_a_count += 1

    # If the number of even A_i is odd, Fennec wins.
    # If the number of even A_i is even, Snuke wins.
    if even_a_count % 2 == 1:
        print("Fennec")
    else:
        print("Snuke")

# Call the solve function to run the game logic.
solve()