import sys
sys.setrecursionlimit(1 << 20)

def main() -> None:
    n = int(sys.stdin.readline().strip())

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def cost(x: int) -> int:
        """
        Total amount of money needed to break the integer x down to all 1's.
        """
        if x <= 1:               # nothing to do for 0 or 1
            return 0
        a = x // 2               # ⌊x/2⌋
        b = x - a                # ⌈x/2⌉   (same as (x + 1) // 2)
        return x + cost(a) + cost(b)

    print(cost(n))

if __name__ == "__main__":
    main()