import sys

def solve():
    """
    Reads card data, determines which cards remain after the discard process,
    and prints the result in the specified format.
    """
    # Use fast I/O
    readline = sys.stdin.readline
    
    try:
        # Read the number of cards
        n_str = readline()
        if not n_str:
            return
        N = int(n_str)
    except (ValueError, IndexError):
        return

    # Store cards as tuples: (Strength, Cost, Original Index)
    cards = []
    for i in range(1, N + 1):
        try:
            A, C = map(int, readline().split())
            cards.append((A, C, i))
        except (ValueError, IndexError):
            # Handle potential empty lines if input format allows
            continue

    # Sort cards by strength (A) in ascending order.
    # Python's default tuple sort uses the first element, which is strength.
    cards.sort()

    # This list will store the indices of the cards that are kept.
    kept_indices = []
    
    # We iterate backwards from the strongest card. A card is kept if it has a
    # lower cost than any stronger card kept so far.
    min_cost_so_far = float('inf')

    # Iterate from the strongest card (last in the sorted list) to the weakest.
    for i in range(N - 1, -1, -1):
        strength, cost, original_index = cards[i]
        
        # A card is kept if no stronger card is also cheaper.
        # `min_cost_so_far` tracks the minimum cost among the stronger cards
        # that have been determined to be kept.
        # If the current card's cost is lower, it becomes a new "skyline" point
        # and is also kept.
        if cost < min_cost_so_far:
            kept_indices.append(original_index)
            # Update the minimum cost seen so far.
            min_cost_so_far = cost
            
    # The indices were added in reverse order of strength; sort them numerically.
    kept_indices.sort()
    
    # Print the count of remaining cards.
    print(len(kept_indices))
    
    # Print the sorted indices of the remaining cards.
    print(*kept_indices)

# Execute the solution
solve()