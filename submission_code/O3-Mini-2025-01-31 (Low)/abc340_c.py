def main():
    import sys
    sys.setrecursionlimit(10**6)
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    N = int(input_data[0])
    
    # We define a recursive function f(x) which returns the total cost paid
    # when the operations start with x.
    # For x >= 2, f(x) = x + f(floor(x/2)) + f(ceil(x/2))
    # and for x < 2 (i.e. x == 1) we need not perform an operation so f(x)=0.
    #
    # Even though the process may produce repeated numbers, notice that the
    # distinct values of x that appear are mostly related by division by 2. This
    # means that there will be at most O(log(N)) distinct values, even for N up to 1e17.
    # We use memoization to avoid recalculating the same values.
    
    memo = {}
    
    def f(x):
        if x < 2:  # When x == 1, no further operation is needed.
            return 0
        if x in memo:
            return memo[x]
        a = x // 2        # floor(x/2)
        b = x - a         # ceil(x/2); note that for x even, a == b, and for x odd, a + 1 == b.
        result = x + f(a) + f(b)
        memo[x] = result
        return result
    
    sys.stdout.write(str(f(N)))
    
if __name__ == '__main__':
    main()