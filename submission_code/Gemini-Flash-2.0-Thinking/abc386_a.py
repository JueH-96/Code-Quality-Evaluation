def solve():
    a, b, c, d = map(int, input().split())
    cards = [a, b, c, d]

    def is_full_house(cards):
        counts = {}
        for card in cards:
            counts[card] = counts.get(card, 0) + 1

        if len(counts) != 2:
            return False

        values = sorted(counts.values())
        return values == [2, 3]

    possible_adds = set(cards)
    found = False
    for add_val in possible_adds:
        five_cards = cards + [add_val]
        if is_full_house(five_cards):
            found = True
            break

    if found:
        print("Yes")
    else:
        print("No")

solve()