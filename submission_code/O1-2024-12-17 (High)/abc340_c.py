def main():
    import sys
    sys.setrecursionlimit(10**7)
    
    N = int(sys.stdin.readline().strip())
    
    memo = {1: 0}  # Base case: F(1) = 0
    
    def F(x):
        if x == 1:
            return 0
        if x in memo:
            return memo[x]
        if x % 2 == 0:
            # x = 2k => F(x) = x + 2 * F(x//2)
            val = x + 2 * F(x // 2)
        else:
            # x = 2k + 1 => F(x) = x + F(x//2) + F(x//2 + 1)
            val = x + F(x // 2) + F(x // 2 + 1)
        memo[x] = val
        return val
    
    print(F(N))

# Call main() at the end
main()