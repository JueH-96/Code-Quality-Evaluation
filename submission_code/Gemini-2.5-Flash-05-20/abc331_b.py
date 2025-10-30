import sys

def solve():
    # Read N, S, M, L from standard input
    line = sys.stdin.readline().split()
    N = int(line[0])
    S = int(line[1])
    M = int(line[2])
    L = int(line[3])

    # Initialize minimum cost to a very large number
    min_cost = float('inf')

    # Determine safe upper bounds for the number of packs.
    # We might need to buy slightly more than N eggs to get a cheaper pack.
    # The maximum excess eggs would be (largest_pack_size - 1), which is 12 - 1 = 11.
    # So, the maximum total eggs we might consider purchasing is N + 11.
    max_eggs_to_consider = N + 11

    # Iterate through possible numbers of 6-egg packs (x)
    # The upper bound for range is (max_eggs_to_consider // 6) + 1 to include the ceiling value.
    for x in range(max_eggs_to_consider // 6 + 1):
        # Iterate through possible numbers of 8-egg packs (y)
        for y in range(max_eggs_to_consider // 8 + 1):
            # Iterate through possible numbers of 12-egg packs (z)
            for z in range(max_eggs_to_consider // 12 + 1):
                
                # Calculate the total number of eggs for the current combination
                total_eggs = 6 * x + 8 * y + 12 * z
                
                # Check if the number of eggs satisfies the requirement
                if total_eggs >= N:
                    # Calculate the cost for the current combination
                    current_cost = x * S + y * M + z * L
                    
                    # Update the minimum cost found so far
                    min_cost = min(min_cost, current_cost)
    
    # Print the minimum cost
    print(min_cost)

# Call the solve function to run the program
solve()