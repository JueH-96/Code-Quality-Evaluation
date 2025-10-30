import sys

def solve():
    # Read N, the number of products
    N = int(sys.stdin.readline())
    
    products = []
    # Read T_i and D_i for each product
    for _ in range(N):
        T, D = map(int, sys.stdin.readline().split())
        S = T  # Start time: product enters range at T_i
        E = T + D  # End time: product leaves range at T_i + D_i
        products.append((S, E))
    
    # Sort products based on their end times (E).
    # If end times are the same, sort by their start times (S).
    # This greedy choice prioritizes products that become unavailable earliest,
    # and among those, prefers ones that are available earlier.
    products.sort(key=lambda x: (x[1], x[0]))
    
    count = 0  # Initialize the count of printed products
    
    # last_print_time tracks the time the most recent product was printed.
    # The printer needs a 1-microsecond cooldown. So, if a print occurs at time 't',
    # the next print can only occur at 't + 1' or later.
    # We initialize it to 0, which means the first print can occur at time max(S, 0 + 1) = max(S, 1).
    # Since T_i >= 1 (and thus S >= 1), this correctly allows the earliest possible print at time 1.
    last_print_time = 0 
    
    # Iterate through the products sorted by their end times
    for S, E in products:
        # Calculate the earliest possible time to print the current product.
        # This time must be:
        # 1. At or after the product's start time (S).
        # 2. At or after the printer becomes available again (last_print_time + 1).
        candidate_print_time = max(S, last_print_time + 1)
        
        # Check if printing at candidate_print_time is possible (i.e., within the product's range)
        if candidate_print_time <= E:
            # If feasible, print this product
            count += 1
            # Update last_print_time to the time this product was printed
            last_print_time = candidate_print_time
            
    # Print the maximum number of products that can be printed
    print(count)

# Call the solve function to execute the program
solve()