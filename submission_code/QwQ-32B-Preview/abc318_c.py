def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    D = int(input[1])
    P = int(input[2])
    F = list(map(int, input[3:3+N]))
    
    # Sort F in descending order
    F.sort(reverse=True)
    
    # Compute total_sum
    total_sum = sum(F)
    
    # Compute prefix_sum
    prefix_sum = [0] * (N + 1)
    for k in range(1, N + 1):
        prefix_sum[k] = prefix_sum[k - 1] + F[k - 1]
    
    # Compute ceil(N/D)
    ceil_N_D = (N + D - 1) // D
    
    # Find minimal cost
    min_cost = float('inf')
    for K in range(0, ceil_N_D + 1):
        if K * D >= N:
            cost = K * P
        else:
            cost = (total_sum - prefix_sum[K * D]) + K * P
        if cost < min_cost:
            min_cost = cost
    
    # Print the minimal cost
    print(min_cost)

if __name__ == "__main__":
    main()