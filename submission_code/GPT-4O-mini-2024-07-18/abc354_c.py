def remaining_cards(n, cards):
    # Sort cards by strength (A_i) and keep track of their original indices
    cards_with_index = sorted((a, c, i + 1) for i, (a, c) in enumerate(cards))
    
    # Initialize a list to hold the remaining cards
    remaining = []
    
    # Track the minimum cost seen so far
    min_cost = float('inf')
    
    # Iterate through the sorted cards
    for a, c, index in cards_with_index:
        # If the current card's cost is less than the minimum cost seen so far,
        # it means this card cannot be discarded
        if c < min_cost:
            remaining.append(index)
            min_cost = c  # Update the minimum cost
    
    # Sort the remaining indices
    remaining.sort()
    
    # Prepare the output
    return (len(remaining), remaining)

import sys
input = sys.stdin.read
data = input().splitlines()

N = int(data[0])
cards = [tuple(map(int, line.split())) for line in data[1:N + 1]]

m, remaining_indices = remaining_cards(N, cards)

# Print the results
print(m)
print(' '.join(map(str, remaining_indices)))