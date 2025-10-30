# YOUR CODE HERE
def find_remaining_cards(N, cards):
    # Sort cards by strength in descending order
    cards.sort(key=lambda x: (-x[1], x[2]))
    
    remaining_cards = []
    min_cost = float('inf')
    
    for card in cards:
        if card[2] < min_cost:
            remaining_cards.append(card[0])
            min_cost = card[2]
    
    remaining_cards.sort()
    return remaining_cards

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
cards = []

for i in range(N):
    A = int(data[2 * i + 1])
    C = int(data[2 * i + 2])
    cards.append((i + 1, A, C))

remaining_cards = find_remaining_cards(N, cards)

print(len(remaining_cards))
print(' '.join(map(str, remaining_cards)))