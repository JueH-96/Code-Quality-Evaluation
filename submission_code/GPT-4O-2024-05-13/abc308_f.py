# YOUR CODE HERE
import heapq

def min_total_cost(N, M, P, L, D):
    items = sorted(P)
    coupons = sorted([(L[i], D[i]) for i in range(M)], reverse=True)
    
    total_cost = 0
    coupon_heap = []
    
    for price in items:
        while coupons and coupons[-1][0] <= price:
            _, discount = coupons.pop()
            heapq.heappush(coupon_heap, -discount)
        
        if coupon_heap:
            total_cost += price + heapq.heappop(coupon_heap)
        else:
            total_cost += price
    
    return total_cost

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
P = list(map(int, data[2:N+2]))
L = list(map(int, data[N+2:N+2+M]))
D = list(map(int, data[N+2+M:]))

print(min_total_cost(N, M, P, L, D))