import sys
from functools import lru_cache

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    
    @lru_cache(maxsize=None)
    def f(x):
        if x < 2:
            return 0
        if x % 2 == 0:
            return x + 2 * f(x // 2)
        else:
            a = x // 2
            return x + f(a) + f(a + 1)
    
    print(f(N))

if __name__ == "__main__":
    main()