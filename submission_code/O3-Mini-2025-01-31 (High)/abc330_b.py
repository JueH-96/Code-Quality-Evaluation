def main():
    import sys

    # Read input from standard input
    data = sys.stdin.read().split()
    if not data:
        return

    # The first three values are N, L, and R
    it = iter(data)
    N = int(next(it))
    L = int(next(it))
    R = int(next(it))
    
    # For each element A_i, clamp it between L and R
    # If A_i is less than L the closest valid value is L,
    # if it's more than R the closest valid value is R,
    # and if it's between L and R, then A_i itself is the closest.
    result = []
    for _ in range(N):
        A_i = int(next(it))
        # Clamp A_i between L and R
        X_i = min(max(L, A_i), R)
        result.append(str(X_i))
    
    # Output results separated by spaces
    sys.stdout.write(" ".join(result))

# Call main to execute the solution
if __name__ == "__main__":
    main()