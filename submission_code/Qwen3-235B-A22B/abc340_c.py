import sys
from functools import lru_cache

sys.setrecursionlimit(1 << 25)

@lru_cache(maxsize=None)
def calculate_cost(n):
    if n < 2:
        return 0
    left = n // 2
    right = (n + 1) // 2
    return n + calculate_cost(left) + calculate_cost(right)

def main():
    N = int(sys.stdin.readline())
    print(calculate_cost(N))

if __name__ == "__main__":
    main()