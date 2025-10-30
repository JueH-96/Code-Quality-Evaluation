import sys

sys.setrecursionlimit(1 << 25)

def main():
    import sys
    N, A, X, Y = map(int, sys.stdin.readline().split())
    memo = {}
    K = 10**6  # Adjust K based on problem constraints and testing

    def e(n):
        if n == 0:
            return 0.0
        if n in memo:
            return memo[n]
        if n > K:
            res = X + e(n // A)
            memo[n] = res
            return res
        # Compute option 1
        option1 = X + e(n // A)
        # Compute option 2
        sum_e = 0.0
        for b in range(1, 7):
            sum_e += e(n // b)
        option2 = Y + sum_e / 6
        res = min(option1, option2)
        memo[n] = res
        return res

    result = e(N)
    print("{0:.12f}".format(result))

if __name__ == '__main__':
    main()