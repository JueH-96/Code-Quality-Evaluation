def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    P = list(map(int, input_data[2:2+N]))
    L = list(map(int, input_data[2+N:2+N+M]))
    D = list(map(int, input_data[2+N+M:]))

    # Pair up coupons (L_i, D_i) and sort them by L_i in ascending order
    coupons = sorted(zip(L, D), key=lambda x: x[0])

    # Sort item prices in descending order
    P.sort(reverse=True)

    import heapq

    total_price = sum(P)
    max_heap = []  # Will store discounts as negative values to simulate max-heap
    idx = 0        # Pointer to coupons in ascending order of L

    # For each item (descending price), add all valid coupons (L <= current price) to the heap
    # Then pick the largest discount from the heap if available.
    for price in P:
        # Add all valid coupons whose L <= price
        while idx < M and coupons[idx][0] <= price:
            # Push the discount as negative
            heapq.heappush(max_heap, -coupons[idx][1])
            idx += 1

        # If we have any coupon in the heap, apply the largest discount
        if max_heap:
            largest_discount = -heapq.heappop(max_heap)
            total_price -= largest_discount

    print(total_price)

# Do not remove the call to the main() function
if __name__ == "__main__":
    main()