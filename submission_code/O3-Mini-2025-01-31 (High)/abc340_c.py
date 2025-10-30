def main():
    import sys
    sys.setrecursionlimit(10000)
    # Read the input (it contains one integer)
    data = sys.stdin.read().split()
    if not data: 
        return
    N = int(data[0])
    
    # Use memoization to speed up the recursive computation
    memo = {1: 0}
    
    # Define the recursive function F(n):
    # F(1) = 0,
    # For n >= 2:
    #   If n is even, then F(n) = n + 2*F(n//2)
    #   If n is odd, then F(n) = n + F(n//2) + F(n//2 + 1)
    def F(n):
        if n in memo:
            return memo[n]
        if n % 2 == 0:
            res = n + 2 * F(n // 2)
        else:
            res = n + F(n // 2) + F(n // 2 + 1)
        memo[n] = res
        return res

    # Compute the total cost for splitting N fully.
    result = F(N)
    sys.stdout.write(str(result))

if __name__ == '__main__':
    main()