import sys

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:                        # no input, nothing to do
        return

    N = int(data[0])                    # number of weeks
    A = list(map(int, data[1:]))        # daily step counts (length should be 7*N)

    weekly_totals = [
        str(sum(A[7 * i: 7 * (i + 1)])) # sum 7 consecutive days for week i
        for i in range(N)
    ]

    print(" ".join(weekly_totals))

if __name__ == "__main__":
    main()