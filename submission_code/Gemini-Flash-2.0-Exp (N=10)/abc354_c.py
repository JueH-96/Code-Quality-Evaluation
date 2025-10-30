def solve():
    n = int(input())
    cards = []
    for i in range(n):
        a, c = map(int, input().split())
        cards.append((a, c, i + 1))

    remaining_cards = list(range(1, n + 1))
    
    while True:
        discarded = False
        for i in range(len(remaining_cards)):
            for j in range(len(remaining_cards)):
                if i == j:
                    continue
                card_x_index = remaining_cards[i] - 1
                card_y_index = remaining_cards[j] - 1
                
                a_x, c_x, _ = cards[card_x_index]
                a_y, c_y, _ = cards[card_y_index]
                
                if a_x > a_y and c_x < c_y:
                    remaining_cards.pop(j)
                    discarded = True
                    break
            if discarded:
                break
        if not discarded:
            break
    
    remaining_cards.sort()
    print(len(remaining_cards))
    print(*remaining_cards)

solve()