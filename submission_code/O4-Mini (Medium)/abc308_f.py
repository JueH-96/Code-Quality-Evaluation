import sys
import threading

def main():
    import sys
    input = sys.stdin.readline
    N, M = map(int, input().split())
    P = list(map(int, input().split()))
    L = list(map(int, input().split()))
    D = list(map(int, input().split()))

    # Pair coupons (L_i, D_i) and sort by L_i
    coupons = sorted(zip(L, D))
    # Sort item prices
    P.sort()

    import heapq
    max_heap = []
    j = 0  # pointer over coupons
    total_cost = sum(P)
    # We'll subtract the best discounts we can apply
    ans = total_cost

    for price in P:
        # Add all coupons whose L_i <= current price
        while j < M and coupons[j][0] <= price:
            # use negative to simulate max-heap
            heapq.heappush(max_heap, -coupons[j][1])
            j += 1
        # If there's any available coupon, use the one with max discount
        if max_heap:
            best_discount = -heapq.heappop(max_heap)
            ans -= best_discount

    print(ans)

if __name__ == "__main__":
    main()