import sys

def calculate_min_cost(N, D, P, F):
    """
    Calculate the minimum possible total cost for the N-day trip.

    Args:
    N (int): The number of days for the trip.
    D (int): The number of one-day passes in a batch.
    P (int): The price of a batch of one-day passes.
    F (list): A list of regular fares for each day.

    Returns:
    int: The minimum possible total cost for the N-day trip.
    """
    min_cost = sum(F)  # Initialize the minimum cost as the sum of all regular fares

    # Calculate the cost for each possible number of batches
    for batches in range(N // D + 1):
        # Calculate the number of one-day passes
        passes = batches * D

        # Calculate the cost of one-day passes
        pass_cost = batches * P

        # Calculate the cost of regular fares for the remaining days
        regular_cost = sum(sorted(F, reverse=True)[passes:])

        # Update the minimum cost
        min_cost = min(min_cost, pass_cost + regular_cost)

    return min_cost

def main():
    # Read the input from stdin
    N, D, P = map(int, sys.stdin.readline().split())
    F = list(map(int, sys.stdin.readline().split()))

    # Calculate and print the minimum possible total cost
    min_cost = calculate_min_cost(N, D, P, F)
    print(min_cost)

if __name__ == "__main__":
    main()