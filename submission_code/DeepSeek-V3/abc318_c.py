# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    D = int(data[1])
    P = int(data[2])
    F = list(map(int, data[3:3+N]))
    
    # Sort F in descending order to prioritize the most expensive days
    F_sorted = sorted(F, reverse=True)
    
    # Precompute the prefix sum of the sorted F
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i+1] = prefix_sum[i] + F_sorted[i]
    
    # Initialize the minimum cost as the sum of all F_i
    min_cost = prefix_sum[N]
    
    # Iterate over possible number of batches
    # The maximum number of batches is ceil(total_days / D)
    # But since P can be very large, we need to find the optimal k
    # We can limit k to the number of days where using a batch is beneficial
    # i.e., k * P < sum of top k*D F_i
    
    # To find the optimal k, we can iterate k from 0 to the maximum possible
    # The maximum k is when k * D >= N, but since P can be large, we need to find the minimum k where k * P is less than the sum of top k*D F_i
    
    # Alternatively, we can find the k where the cost is minimized
    # We can iterate k from 0 to the maximum possible k where k * D <= N + D
    
    # To limit the number of iterations, we can find the k where the cost starts to increase
    # So we can iterate until the cost starts to increase
    
    # Initialize k to 0
    k = 0
    while True:
        # Calculate the number of days covered by k batches
        days_covered = k * D
        # Calculate the cost of using k batches
        cost_batches = k * P
        # Calculate the cost of the remaining days
        cost_remaining = prefix_sum[N] - prefix_sum[min(days_covered, N)]
        # Total cost
        total_cost = cost_batches + cost_remaining
        # Update the minimum cost
        if total_cost < min_cost:
            min_cost = total_cost
        else:
            # If the cost starts to increase, we can break
            break
        # Increment k
        k += 1
        # If k * D exceeds N + D, break
        if k * D > N + D:
            break
    
    print(min_cost)

if __name__ == "__main__":
    main()