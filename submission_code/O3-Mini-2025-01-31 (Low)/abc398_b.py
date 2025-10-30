def main():
    import sys
    cards = list(map(int, sys.stdin.read().strip().split()))
    
    # Count the frequency of each card value
    counts = {}
    for card in cards:
        counts[card] = counts.get(card, 0) + 1

    # We need to check if there is any pair of distinct cards a and b such that:
    # counts[a] >= 3 and counts[b] >= 2
    possible = False
    for a in counts:
        if counts[a] >= 3:
            for b in counts:
                if a != b and counts[b] >= 2:
                    possible = True
                    break
            if possible:
                break
    print("Yes" if possible else "No")

if __name__ == '__main__':
    main()