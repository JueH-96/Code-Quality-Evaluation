def main():
    import sys

    # Read inputs
    data = sys.stdin.read().split()
    A = int(data[0])
    M = int(data[1])
    L = int(data[2])
    R = int(data[3])
    
    # Helper functions for safe integer division
    def floor_div(x, y):
        # Returns floor(x / y) for x,y integers (y>0 guaranteed)
        return x // y

    def ceil_div(x, y):
        # Returns ceil(x / y) for x,y integers (y>0 guaranteed)
        return -((-x) // y)
    
    # We want k such that L <= A + kM <= R
    # => L - A <= kM <= R - A
    # => (L - A)/M <= k <= (R - A)/M
    # k_min = ceil_div(L - A, M)
    # k_max = floor_div(R - A, M)
    
    k_min = ceil_div(L - A, M)
    k_max = floor_div(R - A, M)
    
    # Number of integer k in [k_min, k_max]
    ans = max(0, k_max - k_min + 1)
    
    print(ans)

# Do not forget to call main()
if __name__ == "__main__":
    main()