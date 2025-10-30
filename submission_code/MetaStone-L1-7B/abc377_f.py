def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1

    pieces = []
    R = set()
    C = set()
    S = set()
    D_set = set()

    for _ in range(M):
        x = int(input[idx])
        idx += 1
        y = int(input[idx])
        idx += 1
        pieces.append((x, y))
        R.add(x)
        C.add(y)
        S.add(x + y)
        D_set.add(x - y)

    R = len(R)
    C = len(C)

    sum_diagonals = 0
    for s in S:
        if s < 2:
            continue
        max_i = min(s - 1, 2 * N + 1 - s)
        sum_diagonals += max_i

    sum_anti_diagonals = 0
    for d in D_set:
        max_i = min(d + N, 2 * N - d + 1)
        sum_anti_diagonals += max_i

    overlaps = 0
    for s in S:
        for d in D_set:
            total = s + d
            if total % 2 != 0:
                continue
            i = total // 2
            j = (s - d) // 2
            if 1 <= i <= N and 1 <= j <= N:
                overlaps += 1

    forbidden = (R * N + C * N - R * C) + sum_diagonals + sum_anti_diagonals - overlaps

    available = N * N - forbidden - M
    print(available)

if __name__ == '__main__':
    main()