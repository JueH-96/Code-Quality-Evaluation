def main():
    import sys
    
    # Read inputs
    A, M, L, R = map(int, sys.stdin.read().split())
    
    # We place Christmas trees at positions A + k*M (k is any integer).
    # We want to count how many such positions fall in the interval [L, R].
    #
    # We need k such that L <= A + k*M <= R.
    # Rearrange for k:
    #     L <= A + k*M  -->  k >= (L - A) / M
    #     A + k*M <= R  -->  k <= (R - A) / M
    #
    # Thus k_min = ceil((L - A) / M) and k_max = floor((R - A) / M).
    #
    # Then the number of integers k in [k_min, k_max] is
    #     max(0, k_max - k_min + 1).
    
    # First define a helper to do exact integer ceiling of x / y, for y>0,
    # without using floats (to avoid huge rounding issues).
    # We use the identity ceil(x / y) = - floor((-x) / y).
    def ceil_div(x, y):
        return -((-x) // y)
    
    # Compute k_min and k_max
    k_min = ceil_div(L - A, M)
    k_max = (R - A) // M  # // in Python is floor division, which suits our floor need
    
    # Calculate how many valid k values exist
    if k_max < k_min:
        print(0)
    else:
        print(k_max - k_min + 1)

# Do not forget to call main()
if __name__ == "__main__":
    main()