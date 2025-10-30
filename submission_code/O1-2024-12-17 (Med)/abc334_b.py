def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    A, M, L, R = map(int, data)

    # We want to count integer k with L <= A + k*M <= R.
    # Solve L <= A + k*M => k >= (L - A) / M  (since M>0)
    # and A + k*M <= R => k <= (R - A) / M.

    # Define floor_div and ceil_div to avoid floating point inaccuracies.
    def floor_div(a, b):
        # Returns floor(a/b) for b > 0.
        return a // b

    def ceil_div(a, b):
        # Returns ceil(a/b) for b > 0.
        if a >= 0:
            return (a + b - 1) // b
        else:
            # ceil(a/b) = -floor((-a)/b)
            return -((-a) // b)

    low_k = ceil_div(L - A, M)
    high_k = floor_div(R - A, M)
    result = max(0, high_k - low_k + 1)

    print(result)

# Do not remove: mandatory call to main
if __name__ == "__main__":
    main()