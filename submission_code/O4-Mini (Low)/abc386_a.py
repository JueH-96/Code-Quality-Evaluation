def main():
    import sys
    from collections import Counter

    data = sys.stdin.read().strip().split()
    A, B, C, D = map(int, data)

    # Count the occurrences of each card in the initial four cards
    counts = Counter([A, B, C, D])

    # Try adding one card of value v (1 through 13)
    for v in range(1, 14):
        counts[v] += 1

        # Gather the nonzero counts after adding
        nonzero_counts = [cnt for cnt in counts.values() if cnt > 0]
        nonzero_counts.sort()

        # Check for exactly two distinct values with counts [2, 3]
        if nonzero_counts == [2, 3]:
            print("Yes")
            return

        # Revert the addition before the next trial
        counts[v] -= 1

    # If no addition works, it's impossible
    print("No")

if __name__ == "__main__":
    main()