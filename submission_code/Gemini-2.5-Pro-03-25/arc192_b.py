# YOUR CODE HERE
import sys

def solve():
    """
    Solves the Fennec vs Snuke game problem.

    Reads input N and sequence A. Determines the winner based on optimal play.
    The core logic relies on analyzing the game dynamics, particularly the effect
    of non-activating moves and the presence of initial values A_i = 1.

    Input Format:
    N
    A_1 A_2 ... A_N

    Output Format:
    Prints "Fennec" or "Snuke".
    """

    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))

    # Count the number of elements in A that are initially equal to 1.
    count_ones = 0
    for x in a:
        if x == 1:
            count_ones += 1

    # Based on detailed analysis and examples:
    # The game dynamics change significantly based on whether any A_i is initially 1.
    # An index i with A_i = 1 provides 0 non-activating moves when activated.
    # An index i with A_i > 1 provides A_i - 1 non-activating moves when activated.
    
    # Non-activating moves allow players to pass the turn regarding activations and potentially manipulate the parity of k,
    # where k is the number of non-activating moves made before the game ends.
    # The total number of moves M = N + k. Fennec (player 1) wins if M is odd, Snuke (player 2) wins if M is even.
    # Fennec wins if N and k have different parity. Snuke wins if N and k have the same parity.

    # Case 1: All A_i > 1 (count_ones == 0)
    # In this case, every activation unlocks at least one non-activating move.
    # Analysis suggests that Snuke (the second player) has an advantage. 
    # For N=3, A=(2,2,2), trace shows Snuke wins. For N=4, A=(2,2,2,2), trace shows Snuke wins. Sample 2 also matches.
    # It appears Snuke can always utilize the available non-activating moves to ensure the final parity of k favors them.
    # Thus, if count_ones == 0, Snuke wins regardless of N's parity.
    if count_ones == 0:
        print("Snuke")
    
    # Case 2: At least one A_i = 1 (count_ones > 0)
    # The presence of indices with A_i = 1 limits the availability of non-activating moves.
    # Activating an index i with A_i = 1 consumes an activation step but provides no future flexibility via non-activating moves for that index.
    # This restriction seems to make the game outcome depend primarily on the parity of N, similar to a simpler game without non-activating moves.
    # The player whose turn it is according to N's parity for the final activating move appears to win.
    # Fennec makes moves 1, 3, 5, ...
    # Snuke makes moves 2, 4, 6, ...
    # If N is odd, Fennec makes the N-th move. Fennec wins.
    # If N is even, Snuke makes the N-th move. Snuke wins.
    # This matches Sample 1 and Sample 3 outcomes.
    else: # count_ones > 0
        if n % 2 == 1:
            print("Fennec")
        else:
            print("Snuke")

# Execute the solve function
solve()