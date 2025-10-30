def main():
    import sys, bisect
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    P = int(next(it))
    main_prices = [int(next(it)) for _ in range(N)]
    side_prices = [int(next(it)) for _ in range(M)]
    
    side_prices.sort()
    prefix = [0] * (M + 1)
    for i in range(M):
        prefix[i + 1] = prefix[i] + side_prices[i]

    total = 0
    for a in main_prices:
        # For a fixed main dish with price a:
        # We need side dishes b such that a + b <= P, i.e., b <= P - a.
        x = P - a
        k = bisect.bisect_right(side_prices, x)
        # For these k side dishes, cost is a + b.
        # For the remaining M - k side dishes, cost is P each.
        total += a * k + prefix[k] + (M - k) * P

    sys.stdout.write(str(total))

if __name__ == '__main__':
    main()