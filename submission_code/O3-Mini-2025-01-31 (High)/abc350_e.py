def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().split()
    if not data:
        return
    # Read inputs.
    N = int(data[0])
    A = int(data[1])
    X = int(data[2])
    Y = int(data[3])
    
    # We define f(n) as the minimum expected cost to reduce n to 0.
    # There are two available operations:
    #  1) Pay X and replace n with floor(n/A).
    #  2) Pay Y and roll a die. The die outcome b (from 1 to 6) occurs with probability 1/6.
    #     Then n becomes floor(n/b). In particular, note that when b == 1, the state remains n.
    #
    # If we decide to use the dice option, the expected cost is:
    #   E = Y + (1/6) * [ f(n) + f(n//2) + f(n//3) + f(n//4) + f(n//5) + f(n//6) ].
    # When using dice repeatedly, the decision at state n satisfies:
    #   f(n) = Y + (1/6)*f(n) + (1/6)*[ f(n//2) + f(n//3) + f(n//4) + f(n//5) + f(n//6) ],
    # which can be rearranged to:
    #   (5/6)*f(n) = Y + (1/6)*[ f(n//2) + f(n//3) + f(n//4) + f(n//5) + f(n//6) ],
    # hence
    #   f(n) = (6Y)/5 + (1/5)*[ f(n//2) + f(n//3) + f(n//4) + f(n//5) + f(n//6) ].
    #
    # Thus, we have the recurrence:
    #   f(0) = 0, and for n >= 1:
    #     f(n) = min(  X + f(n//A),
    #                  (6Y)/5 + ( f(n//2) + f(n//3) + f(n//4) + f(n//5) + f(n//6) )/5 )
    
    memo = {}
    def f(n):
        if n == 0:
            return 0.0
        if n in memo:
            return memo[n]
        # Option 1: Operation with fixed divisor A.
        cost1 = X + f(n // A)
        # Option 2: Operation with dice.
        # Here we derive the value from the equation:
        # f(n) = (6Y)/5 + ( f(n//2) + f(n//3) + f(n//4) + f(n//5) + f(n//6) )/5.
        cost_dice = (6 * Y) / 5.0 + ( f(n//2) + f(n//3) + f(n//4) + f(n//5) + f(n//6) ) / 5.0
        res = cost1 if cost1 < cost_dice else cost_dice
        memo[n] = res
        return res

    ans = f(N)
    sys.stdout.write("{:.15f}".format(ans))

if __name__ == '__main__':
    main()