def solve():
    N, D, P = map(int, input().split())
    fares = list(map(int, input().split()))
    
    # Sort fares in descending order
    sorted_fares = sorted(fares, reverse=True)
    
    # Calculate prefix sums for efficient range summing
    prefix_sums = [0]
    for fare in sorted_fares:
        prefix_sums.append(prefix_sums[-1] + fare)
    
    min_total_cost = prefix_sums[N]  # Start with all regular fares
    
    for passes_batches in range(1, (N + D - 1) // D + 1):  # Maximum batches we might need
        pass_cost = passes_batches * P
        passes_days = min(passes_batches * D, N)
        
        # Use passes for the days with the highest fares
        regular_fare_cost = prefix_sums[N] - prefix_sums[passes_days]
        
        total_cost = pass_cost + regular_fare_cost
        
        min_total_cost = min(min_total_cost, total_cost)
    
    return min_total_cost

if __name__ == "__main__":
    print(solve())