def main():
    import sys

    # Read input
    data = list(map(int, sys.stdin.read().strip().split()))
    if not data:
        return
    N, T, P = data[:3]
    L = data[3:]

    # Days needed for each person to reach length T
    days_needed = [max(0, T - length) for length in L]

    # Sort to find the P-th smallest value
    days_needed.sort()

    # The minimal day when at least P people reach T
    answer = days_needed[P - 1]   # 0-indexed

    print(answer)


if __name__ == "__main__":
    main()