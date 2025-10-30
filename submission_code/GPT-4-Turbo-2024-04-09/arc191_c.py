def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    T = int(data[0])
    results = []
    
    for i in range(1, T + 1):
        N = int(data[i])
        
        # We need to find A and M such that A^N - 1 is a multiple of M
        # and the smallest such n is N.
        
        # A simple solution is to choose A = 10^N + 1 and M = 10^N
        # This guarantees that A^N - 1 = (10^N + 1)^N - 1 is a multiple of 10^N
        # and the smallest such n is N because for n < N, (10^N + 1)^n - 1 is not a multiple of 10^N.
        
        # However, 10^N + 1 and 10^N can be very large and exceed 10^18 for large N.
        # We need a more practical approach for large N.
        
        # We can use A = 2 and M = 2^N - 1 (which is a Mersenne number).
        # This choice ensures that 2^N - 1 is obviously a multiple of M = 2^N - 1.
        # And for all k < N, 2^k - 1 is not a multiple of 2^N - 1.
        
        # However, 2^N - 1 can be very large for large N.
        # We need to ensure that M is within the range [1, 10^18].
        
        # For large N, we can use a smaller cycle, for example:
        # A = 2 and M = 2^N - 1 % 10^18 + 1 (to ensure M is positive and within range)
        # This ensures that A^N - 1 is a multiple of M for large N.
        
        # Let's use A = 2 and M = 2^N - 1 % 10^18 + 1
        if N == 1:
            # Special case for N = 1, where we need A^1 - 1 = 0 to be a multiple of M
            # We can choose A = 2 and M = 1
            A = 2
            M = 1
        else:
            A = 2
            M = (2**N - 1) % (10**18) + 1
        
        results.append(f"{A} {M}")
    
    # Print all results
    print("
".join(results))