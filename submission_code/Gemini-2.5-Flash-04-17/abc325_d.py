# YOUR CODE HERE
import sys
import heapq

# Increase recursion depth limit if necessary (usually not for this iterative approach)
# sys.setrecursionlimit(2000)

def solve():
    # Read input using sys.stdin for performance
    N = int(sys.stdin.readline())
    products = []
    for i in range(N):
        T, D = map(int, sys.stdin.readline().split())
        # Store products as (start_time, end_time). End time is T + D.
        products.append((T, T + D))

    # Sort products by their start times (T_i)
    # This allows us to process products in the order they become available
    products.sort()

    count = 0
    # printer_available_time is the earliest time (a specific integer or jumped time point)
    # when the printer is free to potentially print the next product.
    # Initially, the printer is ready at time 0.
    printer_available_time = 0

    # Min-priority queue to store the end times (E_j) of products that have become available
    # (T_j <= current check time) and haven't been printed yet.
    # The priority is the end time, so we can pick the product that finishes earliest.
    # Store only E_j.
    available_pq = [] # min-heap

    # Pointer to the next product to consider from the sorted list by start time
    product_idx = 0

    # Loop while there are products yet to be processed (product_idx < N)
    # or there are products currently available in the priority queue (available_pq).
    while product_idx < N or available_pq:

        # Phase 1: Add newly available products to the priority queue.
        # Consider products whose start time T_i is less than or equal to the current
        # time we are checking for a print opportunity (`printer_available_time`).
        # These products become available at or before the printer *could* be ready.
        while product_idx < N and products[product_idx][0] <= printer_available_time:
            # Add the end time (E_j) of the product to the priority queue.
            # A product is available in [T_i, E_i].
            heapq.heappush(available_pq, products[product_idx][1])
            product_idx += 1

        # Phase 2: Remove expired products from the priority queue.
        # Any product whose end time (E_j) is strictly less than the current time
        # being checked (`printer_available_time`) cannot be printed now or later.
        # Note: A product can be printed *at* its end time.
        while available_pq and available_pq[0] < printer_available_time:
            heapq.heappop(available_pq)

        # Phase 3: Make a print decision or advance time.
        if available_pq:
            # If there are products available at `printer_available_time` (i.e., PQ is not empty
            # after adding new products and removing expired ones), we can perform a print.
            # Greedily print on the product that finishes earliest among those available.
            heapq.heappop(available_pq) # Print on this product (its end time is removed from PQ)
            count += 1
            # Printing takes 1 microsecond charge time. If a print occurs at `printer_available_time`,
            # the printer becomes available again at `printer_available_time + 1`.
            # Set the time for the *next* potential print check.
            printer_available_time += 1
        else:
            # No product is available at `printer_available_time` (after considering new arrivals
            # and removing expired ones).
            # The printer must wait. The next opportunity to potentially print occurs when
            # a new product becomes available.
            # If there are no more products in the sorted list (product_idx == N),
            # and the PQ is empty, then there are no more opportunities, so we break.
            if product_idx < N:
                # The earliest start time of a product not yet considered is `products[product_idx][0]`.
                # Since the PQ was empty, no product starting <= `printer_available_time` was available.
                # The printer must wait until at least `products[product_idx][0]` for a new product.
                # Set the next time to check for a print opportunity to `products[product_idx][0]`.
                # This jumps `printer_available_time` forward, skipping idle time.
                printer_available_time = products[product_idx][0]
            else:
                # No more products to consider in the sorted list, and no products available now.
                break # Loop terminates.

    print(count)

# Standard execution entry point
if __name__ == "__main__":
    solve()