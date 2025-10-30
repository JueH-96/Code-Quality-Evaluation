def main():
    import sys
    from collections import Counter

    cards = list(map(int, sys.stdin.read().split()))
    # We should have exactly 7 cards
    if len(cards) != 7:
        # In case input has extra whitespace or lines, we trim or ignore extras
        cards = cards[:7]

    freq = Counter(cards)
    # Check for any rank x with at least 3 cards
    # paired with a different rank y with at least 2 cards
    for x, cnt_x in freq.items():
        if cnt_x >= 3:
            for y, cnt_y in freq.items():
                if y != x and cnt_y >= 2:
                    print("Yes")
                    return
    print("No")

if __name__ == "__main__":
    main()