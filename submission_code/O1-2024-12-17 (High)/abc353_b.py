def main():
    import sys

    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    A = list(map(int, data[2:]))

    starts = 0
    empty_seats = K

    while A:
        if empty_seats < A[0]:
            # Not enough seats for the front group, so start the attraction.
            starts += 1
            empty_seats = K
        else:
            # Seat the entire front group.
            empty_seats -= A[0]
            A.pop(0)

    # After all groups are seated, we start the attraction one last time.
    starts += 1

    print(starts)

# Do not forget to call main()!
if __name__ == "__main__":
    main()