import sys

# Read input
N = int(sys.stdin.readline().strip())
cards = []
for _ in range(N):
    a, c = map(int, sys.stdin.readline().strip().split())
    cards.append((a, c, i+1))

# Sort cards by strength in descending order and cost in ascending order
cards.sort(key=lambda x: (-x[0], x[1]))

# Discard cards
remaining_cards = []
for a, c, i in cards:
    if not remaining_cards or a > remaining_cards[-1][0] and c < remaining_cards[-1][1]:
        remaining_cards.append((a, c, i))

# Print output
print(len(remaining_cards))
print(" ".join(map(str, [i for _, _, i in remaining_cards])))