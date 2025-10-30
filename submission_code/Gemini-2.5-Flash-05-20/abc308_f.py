# YOUR CODE HERE
import sys
import heapq

def solve():
    # Read N (number of items) and M (number of coupons)
    N, M = map(int, sys.stdin.readline().split())

    # Read item prices P
    P = list(map(int, sys.stdin.readline().split()))

    # Read coupon minimum prices L
    L = list(map(int, sys.stdin.readline().split()))

    # Read coupon discounts D
    D = list(map(int, sys.stdin.readline().split()))

    # Calculate initial total cost by summing all item prices.
    # This is the cost if no coupons are used. Discounts will be subtracted from this.
    total_cost = sum(P)

    # Combine L and D into a list of coupon tuples (L_i, D_i).
    # This makes sorting and accessing coupon properties more convenient.
    coupons = []
    for i in range(M):
        coupons.append((L[i], D[i]))

    # Sort item prices P in ascending order.
    # Processing items from cheapest to most expensive ensures that
    # coupons are considered for the least expensive item they can be applied to.
    P.sort()

    # Sort coupons by their minimum price L_i in ascending order.
    # This allows us to efficiently add coupons to the heap as items become expensive enough.
    coupons.sort()

    # Initialize a max-heap to store available discounts.
    # Python's `heapq` module implements a min-heap. To simulate a max-heap,
    # we store the negative of the discount values. The smallest negative value
    # corresponds to the largest positive discount.
    eligible_discounts_heap = [] # Stores -D_i values

    # `coupon_idx` acts as a pointer to the next coupon to consider from the sorted `coupons` list.
    coupon_idx = 0
    
    # Iterate through each item price, from cheapest to most expensive.
    for item_price in P:
        # Step 1: Add all newly eligible coupons to the heap.
        # A coupon is eligible if its minimum price requirement (L_i) is met by the current `item_price`.
        # Since both `P` and `coupons` are sorted, we can efficiently add coupons in a single pass.
        # Once a coupon becomes eligible, it remains eligible for any subsequent (and thus more expensive) items.
        while coupon_idx < M and coupons[coupon_idx][0] <= item_price:
            # Push the negative discount value onto the heap.
            heapq.heappush(eligible_discounts_heap, -coupons[coupon_idx][1])
            coupon_idx += 1
        
        # Step 2: If there are any eligible coupons in the heap, apply the best one (largest discount).
        if eligible_discounts_heap:
            # Pop the smallest negative value from the heap, which corresponds to the largest positive discount.
            best_discount = -heapq.heappop(eligible_discounts_heap)
            
            # Subtract this best discount from the total cost.
            # This action signifies that we have used this coupon for the current item.
            total_cost -= best_discount

    # Print the final minimum total cost.
    print(total_cost)

# Call the solve function to execute the program.
solve()