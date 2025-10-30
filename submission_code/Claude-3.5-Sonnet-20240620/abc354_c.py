# YOUR CODE HERE
n = int(input())
cards = []
for i in range(n):
    a, c = map(int, input().split())
    cards.append((a, c, i+1))

cards.sort(key=lambda x: (-x[0], x[1]))

remaining = [cards[0]]
for card in cards[1:]:
    if card[1] > remaining[-1][1]:
        remaining.append(card)

remaining.sort(key=lambda x: x[2])

print(len(remaining))
print(' '.join(str(card[2]) for card in remaining))