import sys
import heapq

def solve():
    """
    Reads input, solves the coupon problem, and prints the minimum cost.
    """
    # Read problem dimensions from standard input
    try:
        n, m = map(int, sys.stdin.readline().split())
        
        # Read item prices
        prices = list(map(int, sys.stdin.readline().split()))
        
        # Read coupon thresholds and discounts
        thresholds = list(map(int, sys.stdin.readline().split()))
        discounts = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        # Gracefully exit on malformed or empty input
        return

    # The initial total cost is the sum of all regular prices.
    # The final minimum cost will be this sum minus the maximum possible discount.
    total_cost = sum(prices)

    # Sort items by price in ascending order to process the most constrained items first.
    prices.sort()
    
    # Pair each coupon's threshold with its discount and sort by threshold (ascending).
    # This allows for efficient processing.
    coupons = sorted(zip(thresholds, discounts))

    # A max-heap to store discounts of all currently available coupons.
    # Python's `heapq` is a min-heap, so we store negative values to simulate a max-heap.
    available_discounts_heap = []
    
    coupon_ptr = 0
    total_discount = 0

    # Iterate through each item, from cheapest to most expensive.
    for price in prices:
        # Add all coupons that become eligible for this item to the heap.
        # A coupon is eligible if its price threshold is <= the item's price.
        while coupon_ptr < m and coupons[coupon_ptr][0] <= price:
            discount = coupons[coupon_ptr][1]
            heapq.heappush(available_discounts_heap, -discount)
            coupon_ptr += 1

        # If there are any eligible coupons, apply the one with the highest discount
        # to the current item to maximize the total discount.
        if available_discounts_heap:
            # Pop the largest discount (which is the smallest negative value).
            best_discount = -heapq.heappop(available_discounts_heap)
            total_discount += best_discount

    # The minimum cost is the initial total cost minus the total discount achieved.
    min_cost = total_cost - total_discount
    print(min_cost)

# Execute the solution
solve()