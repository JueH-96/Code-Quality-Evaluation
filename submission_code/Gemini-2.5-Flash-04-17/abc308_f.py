import sys
import heapq

# Function to solve the problem
def solve():
    # Read N and M
    N, M = map(int, sys.stdin.readline().split())

    # Read item prices
    # Use sys.stdin.readline and split for faster input reading
    P = list(map(int, sys.stdin.readline().split()))

    # Read coupon requirements L and discounts D
    L = list(map(int, sys.stdin.readline().split()))
    D = list(map(int, sys.stdin.readline().split()))

    # Calculate initial total cost (sum of regular prices)
    # Python's arbitrary precision integers handle large sums up to 2e14 easily
    total_cost = sum(P)

    # Sort item prices in descending order
    # Processing items from most expensive to cheapest
    sorted_P = sorted(P, reverse=True)

    # Create coupon (L, D) pairs
    coupons = list(zip(L, D))
    # Sort coupons in ascending order based on L
    # This allows us to efficiently add coupons to the heap
    # as we iterate through items in descending price order.
    sorted_coupons = sorted(coupons)

    # Max-heap to store available discounts.
    # This heap contains discounts of coupons whose requirement L
    # is less than or equal to the current item's price being considered.
    # These coupons are available to be used on the current item or any subsequent (cheaper) items.
    # We store negative discounts to use heapq as a max-heap.
    available_discounts_heap = []

    # Pointer for the sorted_coupons list
    # It tracks the index of the next coupon to potentially add to the heap
    coupon_ptr = 0

    # Iterate through sorted item prices from highest to lowest
    for p in sorted_P:
        # Add coupons whose minimum price requirement L is met by the current item price p.
        # Since coupons are sorted by L ascending, we add them as the item price threshold `p`
        # goes down. All coupons with L <= p become eligible when we process an item with price p.
        # Coupons with L <= p will also be eligible for any previous item with price >= p.
        # We only add coupons whose L <= p. The coupon_ptr ensures we only consider
        # coupons with L >= the last added coupon's L.
        while coupon_ptr < M and sorted_coupons[coupon_ptr][0] <= p:
            discount = sorted_coupons[coupon_ptr][1]
            # Add the discount to the heap (as negative for max-heap)
            heapq.heappush(available_discounts_heap, -discount)
            coupon_ptr += 1

        # If there are eligible coupons available for the current item (and potentially cheaper items),
        # use the one with the maximum discount.
        # By using the max discount on the current item (which is the most expensive among remaining),
        # we prioritize getting the largest saving on the item where the coupon is most likely eligible
        # (higher price items are eligible for more coupons).
        # Since D_i >= 1, the discount is always beneficial if applied.
        if available_discounts_heap:
            # Get the maximum available discount from the heap
            max_discount = -heapq.heappop(available_discounts_heap)

            # Apply the discount to the total cost
            total_cost -= max_discount

    # Print the minimum possible total amount
    print(total_cost)

# Call the solve function to run the program
solve()