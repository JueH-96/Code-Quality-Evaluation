def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    groups = list(map(int, data[2:]))

    starts = 0
    empty_seats = K

    for g in groups:
        # If the group doesn't fit, start the attraction and reset seats
        if g > empty_seats:
            starts += 1
            empty_seats = K
        # Seat the group
        empty_seats -= g

    # Once all groups are seated, start the attraction one final time
    starts += 1

    print(starts)

# Don't forget to call main()
if __name__ == "__main__":
    main()