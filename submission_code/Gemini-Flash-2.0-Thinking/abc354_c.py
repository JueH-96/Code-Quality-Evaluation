def solve():
    n = int(input())
    cards_data = []
    for i in range(n):
        a, c = map(int, input().split())
        cards_data.append((a, c, i + 1))

    remaining_cards = set(range(1, n + 1))

    while True:
        discarded_in_iteration = set()
        for y_index in sorted(list(remaining_cards)):
            for x_index in sorted(list(remaining_cards)):
                if x_index != y_index:
                    card_x = next(card for card in cards_data if card[2] == x_index)
                    card_y = next(card for card in cards_data if card[2] == y_index)
                    if card_x[0] > card_y[0] and card_x[1] < card_y[1]:
                        discarded_in_iteration.add(y_index)
                        break
            if y_index in discarded_in_iteration:
                continue

        if not discarded_in_iteration:
            break

        remaining_cards -= discarded_in_iteration

    remaining_cards_list = sorted(list(remaining_cards))
    print(len(remaining_cards_list))
    print(*remaining_cards_list)

solve()