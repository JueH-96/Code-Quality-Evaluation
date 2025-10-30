import sys

def main() -> None:
    # Read input
    M, D = map(int, sys.stdin.readline().split())
    y, m, d = map(int, sys.stdin.readline().split())

    # Move to next day
    if d < D:
        # Just advance the day
        print(y, m, d + 1)
    else:  # d == D, end of the month
        d_next = 1
        if m < M:
            # Advance the month within the same year
            print(y, m + 1, d_next)
        else:  # m == M, end of the year
            print(y + 1, 1, d_next)

if __name__ == "__main__":
    main()