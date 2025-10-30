import sys
from functools import lru_cache

def solve():
    import sys
    sys.setrecursionlimit(1000000)
    N = int(sys.stdin.read())

    @lru_cache(maxsize=None)
    def f(x):
        if x < 2:
            return 0
        if x % 2 == 0:
            return x + 2 * f(x // 2)
        else:
            return x + f(x // 2) + f(x // 2 + 1)

    print(f(N))

if __name__ == "__main__":
    solve()