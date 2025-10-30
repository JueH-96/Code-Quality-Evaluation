# YOUR CODE HERE
import sys

# Define the main logic within a function for clarity
def solve():
    # Read inputs: N = target minimum number of eggs
    # S = cost of a 6-egg pack
    # M = cost of an 8-egg pack
    # L = cost of a 12-egg pack
    N, S, M, L = map(int, sys.stdin.readline().split())

    # Initialize minimum cost to positive infinity. 
    # Using float('inf') is a standard way to represent a very large value.
    min_cost = float('inf')

    # Set a safe upper limit for the number of packs of type 6 and type 8 to iterate through.
    # The problem constraints state N <= 100. 
    # Theoretical analysis shows that the optimal number of packs of each type is relatively small.
    # For example, the maximum number of 6-egg packs needed is roughly ceil((N+11)/6), which is <= 19 for N=100.
    # Similarly, max 8-egg packs is <= 14, and max 12-egg packs is <= 10.
    # Iterating up to 100 packs for 6-egg and 8-egg types (limit=101 means checking 0 to 100 packs) 
    # is definitely safe and computationally feasible. The total number of iterations for the nested loops 
    # will be around 101*101 which is about 10^4, well within typical time limits.
    limit = 101 

    # Iterate through all possible numbers of 6-egg packs (a) from 0 up to limit-1
    for a in range(limit):
        # Iterate through all possible numbers of 8-egg packs (b) from 0 up to limit-1
        for b in range(limit):
            # Calculate the total number of eggs obtained from 'a' 6-egg packs and 'b' 8-egg packs
            eggs_from_6_and_8 = 6 * a + 8 * b
            
            # Calculate the combined cost of these 'a' 6-egg packs and 'b' 8-egg packs
            cost_for_6_and_8 = a * S + b * M
            
            # Calculate how many additional eggs are needed to meet the target N
            # This could be negative if we already have enough eggs from 6 and 8 packs.
            remaining_eggs_needed = N - eggs_from_6_and_8
            
            # Determine the minimum number of 12-egg packs (c) required to cover the remaining need
            c = 0
            # If remaining_eggs_needed is positive, it means we still need more eggs.
            if remaining_eggs_needed > 0:
                # Calculate the ceiling of remaining_eggs_needed / 12 using integer division.
                # This gives the minimum number of 12-egg packs required to get at least remaining_eggs_needed eggs.
                # The formula (X + Y - 1) // Y computes ceil(X / Y) for positive integers X, Y.
                c = (remaining_eggs_needed + 11) // 12
            # If remaining_eggs_needed <= 0, we already met the target N with 6-egg and 8-egg packs,
            # so we don't need any 12-egg packs (c remains 0).

            # Calculate the total cost for this specific combination of (a, b, c) packs
            # Note that 'c' here is the minimum required number of 12-packs for the chosen 'a' and 'b'.
            current_total_cost = cost_for_6_and_8 + c * L
            
            # Update the overall minimum cost found so far if the current combination yields a lower cost.
            min_cost = min(min_cost, current_total_cost)

    # After checking all relevant combinations of 'a' and 'b' (and calculating the optimal 'c' for each pair),
    # print the overall minimum cost found. Since costs are integers, the final min_cost will be an integer.
    # If min_cost remained float('inf'), it would mean no combination reaches N eggs, but constraints guarantee N>=1 and pack sizes are positive, so a solution always exists.
    print(min_cost)

# Execute the solve function to run the program logic
solve()