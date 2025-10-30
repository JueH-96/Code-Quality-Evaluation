def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    existing = set()
    for _ in range(M):
        a = int(input[ptr])
        ptr += 1
        b = int(input[ptr])
        ptr += 1
        existing.add((a, b))

    deltas = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
    attacked = set()

    for a, b in existing:
        for dx, dy in deltas:
            x = a + dx
            y = b + dy
            if 1 <= x <= N and 1 <= y <= N:
                if (x, y) not in existing:
                    attacked.add((x, y))

    total = N * N - M - len(attacked)
    print(total)

if __name__ == "__main__":
    main()