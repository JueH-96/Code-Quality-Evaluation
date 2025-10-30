import sys

def solve():
    N = int(sys.stdin.readline())
    cards = []
    for i in range(N):
        A, C = map(int, sys.stdin.readline().split())
        # Store (A, C, original_index)
        cards.append((A, C, i + 1))

    # Sort cards by strength A in ascending order.
    # The problem guarantees distinct A values.
    # If A values were not distinct, we might need a tie-breaking rule.
    cards.sort()

    # Calculate min_c_strictly_after[i] = min(C_j) for all cards cards[j] with j > i
    # in the sorted list. This is the minimum cost among all cards with strictly greater strength A.
    # Array of size N, indexed 0 to N-1.
    min_c_strictly_after = [float('inf')] * N

    # Iterate backwards from N-2 down to 0
    # min_c_strictly_after[i] = min(cards[i+1][1], min_c_strictly_after[i+1])
    # Base case min_c_strictly_after[N-1] = inf is handled by initialization.
    if N > 1:
        # The minimum cost strictly after the second to last card (index N-2)
        # is the cost of the last card (index N-1), as min(cards[N-1][1], inf) = cards[N-1][1].
        min_c_strictly_after[N-2] = cards[N-1][1]
        
        # Loop from N-3 down to 0
        for i in range(N - 3, -1, -1):
             # The minimum cost strictly after index i is the minimum of the cost at i+1
             # and the minimum cost strictly after i+1.
             min_c_strictly_after[i] = min(cards[i+1][1], min_c_strictly_after[i+1])

    surviving_indices = []
    for i in range(N):
        A_i, C_i, original_idx = cards[i]
        
        # A card y (with attributes A_y, C_y) survives iff there is no card x
        # in the initial set such that A_x > A_y and C_x < C_y.
        # In the sorted list, cards x with A_x > A_y are exactly cards[j] with j > i.
        # So card cards[i] survives iff there is no card cards[j] with j > i
        # such that cards[j][1] < C_i.
        # This is equivalent to C_i <= min(cards[j][1] for j > i).
        # The minimum is min_c_strictly_after[i].
        # So, cards[i] survives iff C_i <= min_c_strictly_after[i].
        # Since costs are distinct, C_i cannot be equal to min_c_strictly_after[i]
        # if min_c_strictly_after[i] is the cost of a different card.
        # If min_c_strictly_after[i] is not inf, it's the cost of some card cards[k] where k > i.
        # If C_i == cards[k][1], it contradicts distinct costs.
        # So if C_i <= min_c_strictly_after[i], it must be C_i < min_c_strictly_after[i].
        
        if C_i < min_c_strictly_after[i]:
            surviving_indices.append(original_idx)

    # Sort the surviving indices in ascending order as required for the output
    surviving_indices.sort()

    # Print the output
    print(len(surviving_indices))
    # Use * to unpack the list elements into the print function, separated by space by default
    if surviving_indices: # Print only if there are surviving cards
        print(*surviving_indices)

solve()