# YOUR CODE HERE
import sys

def solve():
    """
    Reads input, finds the most expensive allowed set meal, and prints its price.
    """
    # Read problem parameters N, M, L.
    try:
        line = sys.stdin.readline()
        if not line:
            return
        N, M, L = map(int, line.split())
    except (ValueError, IndexError):
        # Handles empty input lines locally.
        return

    # Read main dish prices and store as (price, 1-based index) tuples.
    a_prices = list(map(int, sys.stdin.readline().split()))
    main_dishes = [(a_prices[i], i + 1) for i in range(N)]

    # Read side dish prices and store as (price, 1-based index) tuples.
    b_prices = list(map(int, sys.stdin.readline().split()))
    side_dishes = [(b_prices[i], i + 1) for i in range(M)]

    # Read L disallowed pairs and store them in a set for O(1) average lookup.
    disallowed = set()
    for _ in range(L):
        c, d = map(int, sys.stdin.readline().split())
        disallowed.add((c, d))

    # Sort dishes by price in descending order to check most expensive pairs first.
    main_dishes.sort(key=lambda x: x[0], reverse=True)
    side_dishes.sort(key=lambda x: x[0], reverse=True)

    max_price = 0

    # Iterate through each main dish, from most to least expensive.
    for main_price, main_idx in main_dishes:
        # For the current main dish, find the most expensive allowed side dish.
        for side_price, side_idx in side_dishes:
            if (main_idx, side_idx) not in disallowed:
                # Found the best partner for this main dish.
                # This pair's price is a candidate for the overall maximum.
                current_price = main_price + side_price
                max_price = max(max_price, current_price)
                
                # No need to check cheaper side dishes for this main dish.
                break
    
    # Print the final result.
    print(max_price)

solve()