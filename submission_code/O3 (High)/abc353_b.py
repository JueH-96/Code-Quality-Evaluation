import sys

def main() -> None:
    # Read input
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N, K = map(int, data[:2])
    A = list(map(int, data[2:]))

    seats = K          # current empty seats
    starts = 0         # how many times the ride is started

    for g in A:
        # If the current group does not fit, start the ride first
        if seats < g:
            starts += 1
            seats = K
        # Now the group gets on
        seats -= g

    # Start the ride once more after the last group has boarded
    starts += 1

    print(starts)

if __name__ == "__main__":
    main()