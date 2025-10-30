import math

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    P = list(map(int, input[1:N+1]))
    
    # Precompute 0.9^m for all m up to N+1
    pow09 = [1.0] * (N + 2)
    for m in range(1, N + 2):
        pow09[m] = pow09[m-1] * 0.9
    
    # Initialize DP array
    dp = [-math.inf] * (N + 2)
    dp[0] = 0.0  # Base case: sum of 0 elements is 0
    
    # Process each performance value
    for p in P:
        # Update dp in reverse order to prevent overwriting
        for m in range(N, 0, -1):
            if dp[m-1] != -math.inf:
                candidate = dp[m-1] * 0.9 + p
                if candidate > dp[m]:
                    dp[m] = candidate
    
    # Calculate maximum possible R
    max_r = -math.inf
    for m in range(1, N+1):
        if dp[m] == -math.inf:
            continue  # Skip if no valid subset of this size
        Tm = (1.0 - pow09[m]) / 0.1
        term1 = dp[m] / Tm
        term2 = 1200.0 / math.sqrt(m)
        current_r = term1 - term2
        if current_r > max_r:
            max_r = current_r
    
    # Output with sufficient precision
    print("{0:.15f}".format(max_r))

if __name__ == "__main__":
    main()