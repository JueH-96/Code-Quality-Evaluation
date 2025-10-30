def main():
    import sys
    data = sys.stdin.read().split()
    # Parse inputs
    N = int(data[0])
    S = data[1]

    # If there are fewer than 3 seats, no valid i exists
    if N < 3:
        print(0)
        return

    count = 0
    # Check for each i from 1 to N-2 (1-based),
    # which corresponds to indices 0 to N-3 in 0-based
    for i in range(N - 2):
        if S[i] == '#' and S[i + 1] == '.' and S[i + 2] == '#':
            count += 1

    print(count)

if __name__ == "__main__":
    main()