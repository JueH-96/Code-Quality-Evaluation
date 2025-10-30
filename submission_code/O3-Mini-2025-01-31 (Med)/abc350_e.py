def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    A = int(data[1])
    X = int(data[2])
    Y = int(data[3])
    
    from functools import lru_cache
    # Let dp(n) be the minimum expected cost to reduce n to 0.
    #
    # We have two operations:
    #   1. Pay X yen and set n := floor(n/A). This costs X + dp(n//A).
    #   2. Pay Y yen and roll a die:
    #        If outcome is 1, then n stays as n.
    #        If outcome is b (b = 2,3,4,5,6), then n becomes floor(n/b).
    #      Because the roll outcomes are uniform and independent, if we 
    #      commit to using operation 2 repeatedly until progress is made,
    #      its expected cost E satisfies:
    #
    #        E = Y + (1/6)*dp(n) + (1/6)*(dp(n//2)+dp(n//3)+dp(n//4)+dp(n//5)+dp(n//6))
    #
    #      Notice that with outcome 1 the state does not change (n remains n).
    #      Solving for E:
    #          E - (1/6)*dp(n) = Y + (1/6)*sum_{b=2}^{6} dp(n//b)
    #          => E = (6/5)*[ Y + (1/6)*sum_{b=2}^{6} dp(n//b) ]
    #
    # Thus, our recurrence is:
    #   dp(0) = 0
    #   dp(n) = min( X + dp(n//A), (6/5)*( Y + (dp(n//2)+dp(n//3)+dp(n//4)+dp(n//5)+dp(n//6))/6 ) )
    
    @lru_cache(maxsize=None)
    def dp(n):
        if n == 0:
            return 0.0
        # Option 1: Replace n by floor(n/A)
        cost1 = X + dp(n // A)
        
        # Option 2: Roll the die.
        # When rolling the die, outcome 1 means state stays at n.
        # Thus, the expected cost equation becomes:
        #   E = Y + (1/6)*dp(n) + (1/6)*sum_{b=2}^{6} dp(n//b)
        # Which we solve as:  E = (6/5)*( Y + (1/6)*sum_{b=2}^{6} dp(n//b) )
        cost_sum = 0.0
        for b in range(2, 7):
            cost_sum += dp(n // b)
        cost2 = (6.0 / 5.0) * (Y + (cost_sum / 6.0))
        
        return min(cost1, cost2)
    
    # Compute dp for the given N and output the result.
    result = dp(N)
    # Output with sufficient precision.
    sys.stdout.write("{:.15f}".format(result))
    
if __name__ == '__main__':
    main()