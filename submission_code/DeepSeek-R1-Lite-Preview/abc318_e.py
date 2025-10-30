def main():
    import sys
    from collections import defaultdict

    # Read input
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))

    # Group positions by value
    pos = defaultdict(list)
    for idx, val in enumerate(A, 1):
        pos[val].append(idx)

    answer = 0

    for x in pos:
        positions = pos[x]
        c = len(positions)
        if c < 2:
            continue

        # Compute prefix sums
        prefix = [0] * (c + 1)
        for j in range(1, c + 1):
            prefix[j] = prefix[j - 1] + positions[j - 1]

        # Compute sum1: sum over all pairs (pi, pj) of (pj - pi - 1)
        sum1 = 0
        for j in range(2, c + 1):
            sum1 += (j - 1) * positions[j - 1] - prefix[j - 1] - (j - 1)

        # Compute sum2: sum over j=2 to c-1 of (j-1)*(c - j)
        sum2 = 0
        for j in range(2, c - 1 + 1):
            sum2 += (j - 1) * (c - j)

        # Add to answer
        answer += sum1 - sum2

    print(answer)

if __name__ == '__main__':
    main()