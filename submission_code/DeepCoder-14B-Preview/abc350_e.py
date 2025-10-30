def main():
    import sys
    sys.setrecursionlimit(1000000)
    N, A, X, Y = map(int, sys.stdin.readline().split())

    memo = {0: 0.0}

    def E(n):
        if n in memo:
            return memo[n]
        # Compute option1: pay X and divide by A
        option1 = X + E(n // A)
        # Compute option2: pay Y and roll the die
        sum_m = 0
        t = 0
        for b in range(1, 7):
            m = n // b
            if m == n:
                t += 1
            else:
                sum_m += E(m)
        if t == 0:
            option2 = Y + sum_m / 6
        else:
            option2 = (6 * Y + sum_m) / (6 - t)
        res = min(option1, option2)
        memo[n] = res
        return res

    result = E(N)
    print("{0:.12f}".format(result))

if __name__ == '__main__':
    main()