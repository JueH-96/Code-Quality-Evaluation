import sys
import math

def main():
    # Read the input
    N, D, P = map(int, sys.stdin.readline().split())
    F = list(map(int, sys.stdin.readline().split()))

    # Sort the fares in descending order
    F.sort(reverse=True)

    # Calculate the total cost without any passes
    total_cost = sum(F)

    # Calculate the number of passes needed at most
    max_passes_needed = math.ceil(N / D)

    # Try using 0 to max_passes_needed passes to find the minimum cost
    min_cost = total_cost
    for passes_used in range(max_passes_needed + 1):
        # Calculate the cost of the passes
        cost_of_passes = passes_used * P

        # Calculate the cost of the regular fares for the days not covered by passes
        regular_fares_cost = sum(F[passes_used * D:])

        # Calculate the total cost for this number of passes
        current_total_cost = cost_of_passes + regular_fares_cost

        # Update the minimum cost if the current total cost is lower
        min_cost = min(min_cost, current_total_cost)

    # Print the minimum cost
    print(min_cost)

if __name__ == "__main__":
    main()