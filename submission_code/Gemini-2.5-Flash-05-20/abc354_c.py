import sys

def solve():
    N = int(sys.stdin.readline())
    cards = []
    for i in range(N):
        A, C = map(int, sys.stdin.readline().split())
        # Store as (strength, cost, original_1_based_index)
        cards.append((A, C, i + 1))

    # Sort cards by strength (A_i) in descending order.
    # This order ensures that when we process a card (A_y, C_y),
    # any previously processed card (A_x, C_x) will have A_x > A_y.
    cards.sort(key=lambda x: x[0], reverse=True)

    remaining_card_indices = []
    
    # min_C_seen tracks the minimum cost encountered so far among cards
    # that have already been processed (and thus have higher A values than the current card).
    # Initialize with a very large value (infinity) as costs are positive.
    min_C_seen = float('inf') 

    for A, C, original_index in cards:
        # A card (A_y, C_y) remains if there is no other card (A_x, C_x)
        # such that A_x > A_y AND C_x < C_y.
        #
        # Since cards are sorted by A in descending order, for any
        # previously processed card (A_x, C_x), we know A_x > A (current card's strength).
        # So, the current card (A, C) is discarded if its cost C is greater than
        # the minimum cost seen so far (min_C_seen), because that implies there's
        # a previously processed card 'x' with A_x > A AND C_x = min_C_seen < C.
        
        # If the current card's cost C is less than min_C_seen:
        # This card cannot be discarded by any previously processed card,
        # because its cost C is lower than all of them.
        # It also potentially sets a new, lower minimum cost for subsequent (weaker) cards.
        if C < min_C_seen:
            remaining_card_indices.append(original_index)
            min_C_seen = C
        # Else (C > min_C_seen):
        # This means there exists at least one previously processed card 'x'
        # with A_x > A (current strength) and C_x = min_C_seen < C (current cost).
        # Therefore, this current card is discarded by card 'x'.
        # We do not add its index and do not update min_C_seen.

    # The problem requires outputting indices in ascending order.
    remaining_card_indices.sort()

    # Print the count of remaining cards.
    sys.stdout.write(str(len(remaining_card_indices)) + '
')
    
    # Print the sorted indices, space-separated.
    sys.stdout.write(' '.join(map(str, remaining_card_indices)) + '
')

# Call the solve function to run the program.
solve()