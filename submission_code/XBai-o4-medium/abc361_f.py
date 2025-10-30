import sys
from functools import lru_cache

def main():
    N = int(sys.stdin.readline())

    def floor_root(n, b):
        if n == 0:
            return 0
        low = 1
        high = n
        best = 0
        while low <= high:
            mid = (low + high) // 2
            val = 1
            for _ in range(b):
                val *= mid
                if val > n:
                    break
            if val <= n:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        return best

    @lru_cache(maxsize=None)
    def count(n):
        if n < 1:
            return 0
        if n == 1:
            return 1
        res = 0
        max_b = 1
        # Find maximum possible exponent
        while True:
            next_power = 2 ** (max_b + 1)
            if next_power <= n:
                max_b += 1
            else:
                break
        # Iterate from b=2 to max_b
        for b in range(2, max_b + 1):
            s = floor_root(n, b)
            res += s - count(s)
        return res + 1

    print(count(N))

if __name__ == "__main__":
    main()