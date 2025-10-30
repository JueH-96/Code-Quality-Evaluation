def main():
    import sys, math
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return

    # Read input values N, M, K
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])
    
    # Compute the least common multiple (lcm) of N and M.
    gcd_val = math.gcd(N, M)
    lcm = N // gcd_val * M

    # Function to compute the count of numbers ≤ x that are divisible by 
    # exactly one of N and M.
    # The count is:
    #   count = floor(x/N) + floor(x/M) - 2 * floor(x/lcm)
    # because floor(x/lcm) counts numbers divisible by both.
    def count(x):
        return x // N + x // M - 2 * (x // lcm)

    # Binary search to find the smallest x such that count(x) >= K.
    lo = 1
    hi = K * max(N, M)  # This is a safe upper bound.
    
    while lo < hi:
        mid = (lo + hi) // 2
        if count(mid) >= K:
            hi = mid
        else:
            lo = mid + 1

    # Print the found K-th smallest positive integer
    print(lo)

# Call main function as required
if __name__ == "__main__":
    main()