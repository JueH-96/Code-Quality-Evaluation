def main():
    import sys
    sys.setrecursionlimit(1000000)
    input = sys.stdin.readline

    N, A, X, Y = map(int, input().split())
    
    # Use memoization dictionary to avoid recalculations.
    memo = {}

    def f(n):
        # if n becomes 0, no cost
        if n == 0:
            return 0.0
        if n in memo:
            return memo[n]
        
        # Option 1: pay X and do floor division by A.
        cost1 = X + f(n // A)
        
        # Option 2: die roll.
        # Note: when rolling the die, we pay Y and then get an average cost over b=1..6.
        # BUT careful: when b = 1, n // 1 = n; leading to a recurrence f(n)
        # That is, if we directly write:
        #    cost2 = Y + (f(n) + sum_{b=2}^{6} f(n//b)) / 6
        # we need to solve for f(n) when choosing the die roll option.
        # Let E2 = Y + (1/6)*f(n) + (1/6)*sum_{b=2}^{6} f(n//b)
        # Then f(n) = E2, thus f(n) - (1/6)f(n) = Y + (1/6)*sum_{b=2}^{6} f(n//b)
        #  => (5/6) f(n) = Y + (1/6)*sum_{b=2}^{6} f(n//b)
        #  => f(n) = (6/5)*Y + (1/5) * sum_{b=2}^{6} f(n//b)
        # So the effective expected cost from the die roll option is:
        die_sum = 0.0
        for b in range(2, 7):
            die_sum += f(n // b)
        cost2 = (6.0/5.0)*Y + (1.0/5.0)*die_sum
        
        res = min(cost1, cost2)
        memo[n] = res
        return res

    # Compute and print answer with enough precision.
    ans = f(N)
    print("{:.15f}".format(ans))

if __name__ == '__main__':
    main()