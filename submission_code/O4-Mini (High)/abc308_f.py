import sys
import heapq

def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    N = int(next(it)); M = int(next(it))
    # Read item prices
    P = [int(next(it)) for _ in range(N)]
    # Read coupon thresholds and discounts
    L = [int(next(it)) for _ in range(M)]
    D = [int(next(it)) for _ in range(M)]

    # Sort item prices ascending
    P.sort()
    # Pair up (threshold, discount) and sort by threshold
    coupons = list(zip(L, D))
    coupons.sort(key=lambda x: x[0])

    max_heap = []
    total_discount = 0
    j = 0

    # For each item (from cheapest to most expensive),
    # add all newly-eligible coupons into a max‐heap (by discount),
    # then use the best one if available.
    for price in P:
        while j < M and coupons[j][0] <= price:
            # push negative discount for max-heap behavior
            heapq.heappush(max_heap, -coupons[j][1])
            j += 1
        if max_heap:
            total_discount -= heapq.heappop(max_heap)

    # Total cost = sum of all prices minus sum of used discounts
    total_price = sum(P)
    print(total_price - total_discount)

# Call main to execute
main()