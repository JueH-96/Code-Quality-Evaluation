# YOUR CODE HERE
import sys
from heapq import heappush, heappop

def main():
    N, M = map(int, input().split())
    P = list(map(int, input().split()))
    L = list(map(int, input().split()))
    D = list(map(int, input().split()))

    # Sort items by price in descending order
    items = sorted([(p, i) for i, p in enumerate(P)], reverse=True)
    
    # Sort coupons by minimum price in descending order
    coupons = sorted([(l, d) for l, d in zip(L, D)], reverse=True)
    
    coupon_idx = 0
    unused_coupons = []
    total_cost = 0

    for price, item_idx in items:
        # Add eligible coupons to the heap
        while coupon_idx < M and coupons[coupon_idx][0] <= price:
            heappush(unused_coupons, -coupons[coupon_idx][1])
            coupon_idx += 1
        
        # Use the best coupon if available
        if unused_coupons and -unused_coupons[0] <= price:
            discount = -heappop(unused_coupons)
            total_cost += price - discount
        else:
            total_cost += price

    print(total_cost)

if __name__ == "__main__":
    main()