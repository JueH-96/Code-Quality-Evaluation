def main() -> None:
    import sys
    from collections import Counter

    # read four integers
    cards = list(map(int, sys.stdin.read().split()))
    
    # count occurrences of each value
    cnt = Counter(cards)

    # We need to end with exactly two distinct values having counts 3 and 2.
    # If the current number of distinct values is not 2, it is impossible.
    if len(cnt) != 2:
        print("No")
        return

    # Obtain the two counts in ascending order
    counts = sorted(cnt.values())

    # The only workable starting patterns are:
    #  - 3 of one value and 1 of another  -> add to the singleton
    #  - 2 of one value and 2 of another -> add to either
    if counts == [1, 3] or counts == [2, 2]:
        print("Yes")
    else:
        print("No")


if __name__ == "__main__":
    main()