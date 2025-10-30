def main():
    import sys
    data = sys.stdin.read().split()
    N, K = map(int, data[:2])
    A = list(map(int, data[2:]))

    empty = K
    starts = 0

    for group in A:
        if group > empty:
            # start the attraction with current load
            starts += 1
            empty = K
        # guide this group in
        empty -= group

    # after all groups are guided (or couldn't fit), start one final run
    starts += 1

    print(starts)

if __name__ == "__main__":
    main()