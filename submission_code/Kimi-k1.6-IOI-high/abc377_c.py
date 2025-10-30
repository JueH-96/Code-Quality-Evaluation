def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    forbidden = set()
    deltas = [ (2,1), (1,2), (-1,2), (-2,1), (-2,-1), (-1,-2), (1,-2), (2,-1) ]
    for _ in range(M):
        a = int(data[idx])
        idx += 1
        b = int(data[idx])
        idx += 1
        forbidden.add((a, b))
        for di, dj in deltas:
            new_i = a + di
            new_j = b + dj
            if 1 <= new_i <= N and 1 <= new_j <= N:
                forbidden.add((new_i, new_j))
    total = N * N
    ans = total - len(forbidden)
    print(ans)

if __name__ == "__main__":
    main()