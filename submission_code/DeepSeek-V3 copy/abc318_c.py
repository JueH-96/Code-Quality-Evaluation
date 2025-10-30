def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    D = int(data[1])
    P = int(data[2])
    F = list(map(int, data[3:3+N]))
    
    # Sort the fares in descending order
    F_sorted = sorted(F, reverse=True)
    
    # Precompute the prefix sum
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i+1] = prefix_sum[i] + F_sorted[i]
    
    # Initialize the minimum cost as the sum of all fares
    min_cost = sum(F)
    
    # Iterate over possible number of batches
    max_batches = (N // D) + 2
    for k in range(0, max_batches):
        # Calculate the cost for k batches
        cost = k * P
        # The number of days covered by passes is k * D
        covered = k * D
        # The remaining days are N - covered
        # We need to pay for the remaining days, but only the ones not covered by passes
        # Since passes can be used on any day, we can choose the most expensive days to cover
        # So, the cost for the remaining days is the sum of the fares for the days not covered
        # Which is the sum of all fares minus the sum of the top covered fares
        if covered >= N:
            # All days are covered by passes
            cost += 0
        else:
            cost += prefix_sum[N] - prefix_sum[covered]
        # Update the minimum cost
        if cost < min_cost:
            min_cost = cost
    
    print(min_cost)

if __name__ == "__main__":
    main()