def solve(N, cards):
    # Augment cards with their original indices (1-indexed)
    cards_with_idx = [(i+1, A, C) for i, (A, C) in enumerate(cards)]
    
    # Sort cards by strength in ascending order
    cards_with_idx.sort(key=lambda x: x[1])
    
    # From right to left, keep track of the minimum cost seen so far
    min_cost = float("inf")
    remaining_cards = []
    
    for i in range(N - 1, -1, -1):
        idx, A, C = cards_with_idx[i]
        if C < min_cost:
            remaining_cards.append(idx)
            min_cost = C
    
    # Sort the remaining cards by their original index
    remaining_cards.sort()
    
    return remaining_cards

# Input handling
N = int(input())
cards = []
for _ in range(N):
    A, C = map(int, input().split())
    cards.append((A, C))

# Solve the problem
remaining_cards = solve(N, cards)

# Output
print(len(remaining_cards))
print(" ".join(map(str, remaining_cards)))