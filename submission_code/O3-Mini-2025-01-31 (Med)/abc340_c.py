def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    
    # Define a recursive function with memoization.
    # We define F(n) as the total cost when starting with n.
    # Base case: F(1) = 0 because we do not perform any operation on 1.
    # For even n (n = 2m): We have
    #   F(2m) = 2m + 2 * F(m)
    # because the two numbers after splitting are both m.
    # For odd n (n = 2m+1): We have
    #   F(2m+1) = (2m+1) + F(m) + F(m+1)
    # because the two numbers after splitting are m and m+1.
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def F(n):
        if n == 1:
            return 0
        if n % 2 == 0:
            return n + 2 * F(n // 2)
        else:
            return n + F(n // 2) + F((n // 2) + 1)
    
    answer = F(N)
    sys.stdout.write(str(answer))

# Call main function as required.
if __name__ == '__main__':
    main()