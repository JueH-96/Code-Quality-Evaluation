def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N, D, P = map(int, data[:3])
    F = list(map(int, data[3:]))

    # Sort fares in descending order
    F.sort(reverse=True)

    # Prefix sums: prefix[i] = sum of F[0] + F[1] + ... + F[i-1]
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i+1] = prefix[i] + F[i]
    
    # Compute total sum of fares
    total_fares = prefix[N]
    
    min_cost = total_fares  # Case when no passes are bought
    
    # Maximum number of batches to consider
    max_batches = (N + D - 1) // D

    for x in range(max_batches + 1):
        days_covered = x * D
        if days_covered > N:
            days_covered = N
        # Cost = cost of x batches + sum of fares not covered
        cost = x * P + (total_fares - prefix[days_covered])
        if cost < min_cost:
            min_cost = cost
    
    print(min_cost)

# Do not forget to call main()!
main()