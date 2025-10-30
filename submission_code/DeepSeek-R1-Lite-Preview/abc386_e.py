def main():
    import sys

    # Read input
    input = sys.stdin.read().split()
    N = int(input[0])
    K = int(input[1])
    A = list(map(int, input[2:2+N]))

    result = 0
    for bit in range(59, -1, -1):
        # Count numbers with the current bit set
        S = sum(1 for num in A if (num >> bit) & 1)
        T = N - S
        # Calculate lower and upper bounds for x
        lower = max(1, K - T)
        upper = min(S, K)
        # Check if there exists an odd x in [lower, upper]
        if lower <= upper:
            # If lower is odd or upper is odd, there exists an odd x
            if lower % 2 == 1 or upper % 2 == 1:
                # Set this bit in the result
                result |= (1 << bit)
    
    print(result)

if __name__ == "__main__":
    main()