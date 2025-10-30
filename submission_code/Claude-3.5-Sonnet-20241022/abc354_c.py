N = int(input())
cards = []
for i in range(N):
    a, c = map(int, input().split())
    cards.append((a, c, i+1))

# Keep discarding cards until no more discards possible
while True:
    discarded = False
    for i in range(len(cards)):
        for j in range(len(cards)):
            if i != j:
                # If card i has higher strength and lower cost than card j
                if cards[i][0] > cards[j][0] and cards[i][1] < cards[j][1]:
                    # Discard card j
                    cards.pop(j)
                    discarded = True
                    break
        if discarded:
            break
    if not discarded:
        break

# Sort remaining cards by index
remaining = sorted([card[2] for card in cards])

# Print output
print(len(remaining))
print(*remaining)