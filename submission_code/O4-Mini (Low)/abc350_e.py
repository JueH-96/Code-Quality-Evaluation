import sys
sys.setrecursionlimit(10000)

def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = int(data[1])
    X = float(data[2])
    Y = float(data[3])

    from functools import lru_cache

    @lru_cache(None)
    def g(n):
        # Base case
        if n == 0:
            return 0.0
        # Option 1: pay X, go to floor(n/A)
        cost1 = X + g(n // A)
        # Option 2: roll die, algebraically solve out g(n)
        # sum over b=2..6 of g(floor(n/b))
        s = 0.0
        # b=1 would give self-reference, so skip b=1
        for b in range(2, 7):
            s += g(n // b)
        # g2 = (6*Y + s) / 5
        cost2 = (6.0 * Y + s) / 5.0
        return min(cost1, cost2)

    ans = g(N)
    # Print with sufficient precision
    print("{:.10f}".format(ans))

if __name__ == "__main__":
    main()