def main():
    import sys
    data = list(map(int, sys.stdin.read().split()))
    A, B, C, D = data

    from collections import Counter

    # We will try adding one card x in range [1..13].
    # Then check if we can form a Full House.
    # A Full House means: 3 cards of one value and 2 cards of another distinct value.

    cards = [A, B, C, D]

    for x in range(1, 14):
        new_cards = cards + [x]
        freq = Counter(new_cards)
        # We need exactly 2 distinct values with frequencies 3 and 2.
        if len(freq) == 2:
            values = list(freq.values())
            if sorted(values) == [2, 3]:
                print("Yes")
                return

    print("No")

# Do not remove this call
main()