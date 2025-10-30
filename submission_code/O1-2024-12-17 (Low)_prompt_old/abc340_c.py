def solve():
    import sys
    sys.setrecursionlimit(10**7)

    N = int(sys.stdin.readline().strip())

    memo = {1: 0}

    def cost(x):
        # If we already computed cost(x), return it
        if x in memo:
            return memo[x]

        # If x is a power of two, cost(2^m) = m * 2^m
        if (x & (x - 1)) == 0:  # fast check for power of 2
            m = x.bit_length() - 1  # x == 2^m => m = log2(x)
            memo[x] = m * x
            return memo[x]

        if x % 2 == 0:
            # Even case
            ans = x + 2 * cost(x // 2)
        else:
            # Odd case
            ans = x + cost(x // 2) + cost(x // 2 + 1)
        memo[x] = ans
        return ans

    print(cost(N))

# For local testing purposes (comment out if running on an online judge):
# if __name__ == "__main__":
#     solve()