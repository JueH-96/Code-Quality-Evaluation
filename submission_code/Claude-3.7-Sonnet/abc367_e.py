def main():
    # Read input
    N, K = map(int, input().split())
    X = list(map(int, input().split()))
    A = list(map(int, input().split()))
    
    # Convert to 1-indexed for easier handling
    X = [0] + X
    A = [0] + A
    
    # If K is 0, return the original sequence
    if K == 0:
        print(' '.join(map(str, A[1:])))
        return
    
    # Compute the powers of X: X^1, X^2, X^4, X^8, ...
    powers = []
    
    # Start with X^1
    current_power = X.copy()
    powers.append(current_power)
    
    # Compute the rest of the powers
    while (1 << len(powers)) <= K:
        next_power = [0] * (N + 1)
        for j in range(1, N + 1):
            next_power[j] = current_power[current_power[j]]
        current_power = next_power
        powers.append(current_power)
    
    # Compute where each index goes after K operations
    result = list(range(N + 1))
    
    for i in range(len(powers)):
        if (K >> i) & 1:  # Check if the i-th bit of K is set
            next_result = [0] * (N + 1)
            for j in range(1, N + 1):
                next_result[j] = powers[i][result[j]]
            result = next_result
    
    # Compute the final sequence
    final_A = [0] * (N + 1)
    for i in range(1, N + 1):
        final_A[i] = A[result[i]]
    
    # Output the result
    print(' '.join(map(str, final_A[1:])))

if __name__ == "__main__":
    main()