def main():
    import sys
    import bisect

    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    D = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(M)]

    A.sort()
    B.sort()

    ans = -1
    for a in A:
        # We want the largest b such that b <= a + D
        idx = bisect.bisect_right(B, a + D) - 1
        if idx >= 0:
            b = B[idx]
            # Check if it also satisfies b >= a - D
            if b >= a - D:
                s = a + b
                if s > ans:
                    ans = s

    print(ans)

if __name__ == "__main__":
    main()