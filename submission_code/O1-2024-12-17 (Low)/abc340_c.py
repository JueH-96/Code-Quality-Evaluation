def main():
    import sys
    sys.setrecursionlimit(10**7)
    from functools import lru_cache

    N = int(sys.stdin.readline().strip())

    @lru_cache(None)
    def cost(x):
        if x <= 1:
            return 0
        return x + cost(x // 2) + cost((x + 1) // 2)

    print(cost(N))

# Call main
if __name__ == "__main__":
    main()