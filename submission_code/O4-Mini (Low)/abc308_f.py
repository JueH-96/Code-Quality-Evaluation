import sys
import threading

def main():
    import sys
    input = sys.stdin.readline

    N, M = map(int, input().split())
    P = list(map(int, input().split()))
    L = list(map(int, input().split()))
    D = list(map(int, input().split()))

    # Pair coupons as (L_i, D_i) and sort by L ascending
    coupons = sorted(zip(L, D), key=lambda x: x[0])
    # Sort item prices descending
    P.sort(reverse=True)

    import heapq
    max_heap = []
    j = 0
    total_discount = 0

    for price in P:
        # Push into heap all coupons whose L_i <= price
        while j < M and coupons[j][0] <= price:
            # We want max-heap on D_i; Python has min-heap, so push negative
            heapq.heappush(max_heap, -coupons[j][1])
            j += 1
        # If there's any coupon available, take the one with max D_i
        if max_heap:
            best = -heapq.heappop(max_heap)
            total_discount += best

    # The cost is sum of prices minus total_discount
    print(sum(P) - total_discount)

if __name__ == "__main__":
    main()