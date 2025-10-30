def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    directions = [ (2,1), (1,2), (-1,2), (-2,1), (-2,-1), (-1,-2), (1,-2), (2,-1) ]
    s = set()
    for _ in range(M):
        a = int(input[idx])
        idx += 1
        b = int(input[idx])
        idx += 1
        s.add((a, b))
        for di, dj in directions:
            new_i = a + di
            new_j = b + dj
            if 1 <= new_i <= N and 1 <= new_j <= N:
                s.add((new_i, new_j))
    total = N * N
    print(total - len(s))
    
if __name__ == "__main__":
    main()