import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(10000)
    from functools import lru_cache

    N = int(sys.stdin.readline().strip())

    @lru_cache(None)
    def total_cost(x):
        # For x < 2, no operations are performed
        if x < 2:
            return 0
        # Split x into floor(x/2) and ceil(x/2), pay x,
        # then pay recursively for each half.
        a = x // 2
        b = x - a  # this is ceil(x/2)
        return x + total_cost(a) + total_cost(b)

    print(total_cost(N))

if __name__ == "__main__":
    main()