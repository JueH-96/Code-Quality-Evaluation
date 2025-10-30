import sys

def find_remaining_cards(cards):
    # Sort cards by strength in descending order, then by cost in ascending order
    sorted_cards = sorted(cards, key=lambda x: (-x[1], x[2]))
    remaining_cards = []
    min_cost = float('inf')
    
    # Iterate through the sorted cards
    for card in sorted_cards:
        idx, strength, cost = card
        # If the cost of the current card is less than the minimum cost found so far,
        # it cannot be discarded, so we add it to the remaining cards
        if cost < min_cost:
            remaining_cards.append(idx)
            min_cost = cost
    
    # Sort the remaining cards by their original index
    remaining_cards.sort()
    return remaining_cards

# Read input
N = int(sys.stdin.readline().strip())
cards = []
for i in range(N):
    A_i, C_i = map(int, sys.stdin.readline().split())
    cards.append((i + 1, A_i, C_i))

# Find the remaining cards
remaining_cards = find_remaining_cards(cards)

# Write output
print(len(remaining_cards))
print(' '.join(map(str, remaining_cards)))