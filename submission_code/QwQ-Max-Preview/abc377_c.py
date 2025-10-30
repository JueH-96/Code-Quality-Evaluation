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
    
    deltas = [(2, 1), (1, 2), (-1, 2), (-2, 1),
              (-2, -1), (-1, -2), (1, -2), (2, -1)]
    
    forbidden = set()
    for a, b in existing:
        forbidden.add((a, b))
        for da, db in deltas:
            ni = a + da
            nj = b + db
            if 1 <= ni <= N and 1 <= nj <= N:
                forbidden.add((ni, nj))
    
    total = N * N
    answer = total - len(forbidden)
    print(answer)

if __name__ == "__main__":
    main()