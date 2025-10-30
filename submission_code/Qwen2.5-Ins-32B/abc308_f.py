import sys
from heapq import heappush, heappop

def solve():
    input = sys.stdin.read
    N, M, *data = map(int, input().split())
    P = data[:N]
    L = data[N:N+M]
    D = data[N+M:]
    
    # Create a list of tuples (L_i, D_i) and sort it in descending order
    coupons = sorted(zip(L, D), reverse=True)
    
    # Create a min-heap to store the discounts we can use
    discount_heap = []
    total_cost = 0
    
    # Sort items by price in descending order
    for price in sorted(P, reverse=True):
        # Add all applicable coupons to the heap
        while coupons and coupons[-1][0] <= price:
            heappush(discount_heap, -coupons.pop()[1])
        
        # Apply the best available discount if possible
        if discount_heap:
            total_cost += price + heappop(discount_heap)
        else:
            total_cost += price
    
    print(total_cost)

solve()