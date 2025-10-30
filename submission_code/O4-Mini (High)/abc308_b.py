def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    C = [next(it) for _ in range(N)]
    D = [next(it) for _ in range(M)]
    P = [int(next(it)) for _ in range(M+1)]
    default_price = P[0]
    price_map = {D[i]: P[i+1] for i in range(M)}
    total = 0
    for color in C:
        total += price_map.get(color, default_price)
    print(total)

if __name__ == "__main__":
    main()