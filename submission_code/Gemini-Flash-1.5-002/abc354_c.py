# YOUR CODE HERE
n = int(input())
cards = []
for i in range(n):
    a, c = map(int, input().split())
    cards.append((a, c, i + 1))

remaining_cards = set(range(1, n + 1))
while True:
    discarded = False
    for i in range(len(cards)):
        for j in range(len(cards)):
            if i != j:
                a1, c1, idx1 = cards[i]
                a2, c2, idx2 = cards[j]
                if a1 > a2 and c1 < c2 and idx2 in remaining_cards:
                    remaining_cards.remove(idx2)
                    discarded = True
    if not discarded:
        break

remaining_cards_list = sorted(list(remaining_cards))
print(len(remaining_cards_list))
print(*remaining_cards_list)