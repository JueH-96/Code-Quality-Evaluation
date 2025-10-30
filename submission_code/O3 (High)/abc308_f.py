import sys
import heapq

def main() -> None:
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    # --- read input ---------------------------------------------------------
    N, M = map(int, input().split())
    P = list(map(int, input().split()))
    L = list(map(int, input().split()))
    D = list(map(int, input().split()))

    # --- sort items and coupons --------------------------------------------
    prices = sorted(P)                                 # items: low → high
    coupons = sorted(zip(L, D))                       # by threshold L

    # --- sweep over items ---------------------------------------------------
    idx = 0                                           # pointer in coupons
    heap = []                                         # max-heap of discounts
    total_discount = 0

    for price in prices:
        # add all coupons whose threshold ≤ current price
        while idx < M and coupons[idx][0] <= price:
            heapq.heappush(heap, -coupons[idx][1])    # negative => max-heap
            idx += 1

        # use the best (largest) discount that is currently available
        if heap:
            total_discount += -heapq.heappop(heap)

    # --- output answer ------------------------------------------------------
    total_cost = sum(prices) - total_discount
    print(total_cost)

if __name__ == "__main__":
    main()