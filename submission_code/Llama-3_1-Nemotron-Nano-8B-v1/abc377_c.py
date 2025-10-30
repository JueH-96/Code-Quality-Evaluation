def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    existing = set()
    for _ in range(M):
        a = int(input[idx])
        idx += 1
        b = int(input[idx])
        idx += 1
        existing.add((a, b))
    
    S = set()
    positions_to_check = [
        (a + 2, b + 1),
        (a + 1, b + 2),
        (a - 1, b + 2),
        (a - 2, b + 1),
        (a - 2, b - 1),
        (a - 1, b - 2),
        (a + 1, b - 2),
        (a + 2, b - 1),
    ]
    for a, b in existing:
        for x, y in positions_to_check:
            if 1 <= x <= N and 1 <= y <= N:
                S.add((x, y))
    
    forbidden = len(S) - len(S & existing)
    answer = N * N - M - forbidden
    print(answer)

if __name__ == "__main__":
    main()