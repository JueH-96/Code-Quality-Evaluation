# YOUR CODE HERE
import sys
import heapq

# Function to solve the problem
def solve():
    # Read N (number of items) and M (number of coupons) from standard input
    N, M = map(int, sys.stdin.readline().split())
    
    # Read item prices P_1 ... P_N
    P = list(map(int, sys.stdin.readline().split()))
    
    # Read coupon thresholds L_1 ... L_M
    L = list(map(int, sys.stdin.readline().split()))
    
    # Read coupon discounts D_1 ... D_M
    D = list(map(int, sys.stdin.readline().split()))

    # Calculate the initial total cost assuming no coupons are used
    # This is the sum of all regular item prices.
    total_cost = sum(P)

    # Sort the item prices in ascending order.
    # The greedy strategy relies on processing items from cheapest to most expensive.
    P.sort()

    # Create a list of coupons, where each coupon is represented as a tuple (threshold, discount).
    # This pairs up the threshold L_j and discount D_j for each coupon j.
    coupons = []
    for j in range(M):
        coupons.append((L[j], D[j]))
    
    # Sort the coupons based on their threshold L_j in ascending order.
    # This allows us to efficiently find coupons that become eligible as we process items
    # with increasing prices.
    coupons.sort()

    # Initialize a min-heap (priority queue) to store the discounts of eligible coupons.
    # A coupon is eligible for an item if its threshold is less than or equal to the item's price.
    # Since Python's heapq module implements a min-heap, we store negative discounts (-D_j).
    # This allows us to extract the maximum discount by popping the minimum element (most negative value).
    eligible_discounts_heap = []
    
    # Initialize an index pointer for the sorted coupons list. This tracks the next coupon to consider.
    coupon_idx = 0
    
    # Initialize the total discount obtained so far. This will be maximized by the greedy strategy.
    total_discount = 0

    # Iterate through each item, processed in order of increasing price (due to sorting P).
    for i in range(N):
        current_price = P[i]
        
        # Check for any new coupons that become eligible because their threshold L_j
        # is less than or equal to the current item's price.
        # We advance the coupon_idx pointer as long as the condition holds.
        while coupon_idx < M and coupons[coupon_idx][0] <= current_price:
            # Get the discount of the newly eligible coupon.
            discount = coupons[coupon_idx][1]
            # Add the negative discount to the min-heap.
            heapq.heappush(eligible_discounts_heap, -discount)
            # Advance the coupon pointer to consider the next coupon in the sorted list.
            coupon_idx += 1
        
        # After potentially adding new eligible coupons, check if there are any coupons in the heap.
        # If the heap is not empty, it contains discounts of coupons that are eligible for the current item
        # (because their L_j <= current_price) and have not been used yet.
        if eligible_discounts_heap:
            # Greedily select and "use" the coupon with the maximum discount among all currently eligible ones.
            # This corresponds to popping the minimum element (most negative discount) from the min-heap.
            max_discount_neg = heapq.heappop(eligible_discounts_heap)
            # Add the actual discount value (positive) to the total accumulated discount.
            # Each pop corresponds to using one coupon for one item.
            total_discount += -max_discount_neg

    # The minimum possible total amount required to buy all items is the initial total cost
    # minus the maximum total discount achieved by optimally applying coupons.
    print(total_cost - total_discount)

# Call the solve function to execute the logic when the script runs
solve()