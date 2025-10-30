def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    P = int(next(it))
    Q = int(next(it))
    # Read dish prices
    D = [int(next(it)) for _ in range(N)]
    # Option 1: pay regular price P
    ans = P
    # Option 2: use coupon, pay Q + min(D_i)
    for d in D:
        ans = min(ans, Q + d)
    print(ans)

if __name__ == "__main__":
    main()