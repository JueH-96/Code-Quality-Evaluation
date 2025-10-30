import sys
sys.setrecursionlimit(1 << 25)

def main():
    N = int(sys.stdin.readline())
    memo = {}

    def cost(x):
        if x < 2:
            return 0
        if x in memo:
            return memo[x]
        a = x // 2
        b = (x + 1) // 2
        res = x + cost(a) + cost(b)
        memo[x] = res
        return res

    print(cost(N))

if __name__ == "__main__":
    main()