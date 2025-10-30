def main():
    import sys
    from collections import Counter

    data = sys.stdin.read().strip().split()
    if len(data) < 4:
        return
    cards = list(map(int, data))
    base_count = Counter(cards)

    # Try adding one card (from 1 to 13) and check if it forms a Full House.
    # A Full House means the new 5 cards have exactly two distinct numbers,
    # with counts 3 and 2.
    for card in range(1, 14):
        new_count = base_count.copy()
        new_count[card] += 1
        if len(new_count) == 2:
            values = sorted(new_count.values())
            if values == [2, 3]:
                sys.stdout.write("Yes")
                return
    sys.stdout.write("No")

if __name__ == '__main__':
    main()