def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    D = int(data[1])
    P = int(data[2])
    F = list(map(int, data[3:]))
    
    # Calculate the total cost if paying regular fare for all days
    total_regular_cost = sum(F)
    
    # Calculate the minimum cost using passes
    min_cost = total_regular_cost  # Start with the cost of paying all days regularly
    
    # We will try to use k batches of passes where k ranges from 0 to (N + D - 1) // D
    max_batches = (N + D - 1) // D
    
    for k in range(max_batches + 1):
        passes_cost = k * P
        # Calculate the cost of the days not covered by passes
        if k * D >= N:
            # All days can be covered by passes
            cost_with_passes = passes_cost
        else:
            # Some days are not covered by passes
            # We need to pay regular fare for the remaining days
            # Sort the fares and pay for the most expensive days with passes
            sorted_fares = sorted(F, reverse=True)
            cost_with_passes = passes_cost + sum(sorted_fares[k * D:])
        
        # Update the minimum cost found
        min_cost = min(min_cost, cost_with_passes)
    
    print(min_cost)

if __name__ == "__main__":
    main()