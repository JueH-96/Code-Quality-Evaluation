def solve():
    import sys
    N = int(sys.stdin.read())
    for n in range(N, 1000):
        h = n // 100
        t = (n // 10) % 10
        o = n % 10
        if h * t == o:
            print(n)
            return

solve()