import sys
from functools import lru_cache

def main():
    N = int(sys.stdin.readline())
    
    @lru_cache(maxsize=None)
    def calculate(x):
        if x < 2:
            return 0
        if x % 2 == 0:
            half = x // 2
            return x + 2 * calculate(half)
        else:
            a = x // 2
            b = a + 1
            return x + calculate(a) + calculate(b)
    
    print(calculate(N))

if __name__ == "__main__":
    main()