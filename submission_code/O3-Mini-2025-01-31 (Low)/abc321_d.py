def main():
    import sys, bisect
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    P = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(M)]
    
    B.sort()
    prefix = [0] * (M + 1)
    for i in range(M):
        prefix[i + 1] = prefix[i] + B[i]
    
    total = 0
    for a in A:
        # For current main dish with price a,
        # we want to choose all side dishes B_j such that a + B_j <= P.
        # Let X = P - a. All B_j <= X will contribute as (a + B_j).
        X = P - a
        idx = bisect.bisect_right(B, X)
        # For the first idx dishes: set meal price = a + B_j
        total += a * idx + prefix[idx]
        # For the remaining dishes: set meal price = P.
        total += (M - idx) * P
    
    sys.stdout.write(str(total))

if __name__ == '__main__':
    main()