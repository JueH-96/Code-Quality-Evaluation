# YOUR CODE HERE
def solve():
    n = int(input())
    cards = []
    for i in range(n):
        a, c = map(int, input().split())
        cards.append((a, c, i + 1))

    remaining_cards = list(range(1, n + 1))

    while True:
        discarded = False
        for i in range(len(cards)):
            for j in range(len(cards)):
                if i != j:
                    if cards[i][0] > cards[j][0] and cards[i][1] < cards[j][1]:
                        if cards[j][2] in remaining_cards:
                            remaining_cards.remove(cards[j][2])
                            discarded = True
        
        if not discarded:
            break
        
        new_cards = []
        for card in cards:
            if card[2] in remaining_cards:
                new_cards.append(card)
        cards = new_cards

    remaining_cards.sort()
    print(len(remaining_cards))
    print(*remaining_cards)


solve()