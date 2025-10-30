import heapq
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    
    P = list(map(int, data[2:N+2]))
    L = list(map(int, data[N+2:N+2+M]))
    D = list(map(int, data[N+2+M:]))
    
    # Sort items by price
    items = sorted(P)
    
    # Create a list of (L_i, D_i) and sort by L_i
    coupons = sorted(zip(L, D))
    
    # Minimize total cost
    total_cost = 0
    coupon_index = 0
    max_heap = []
    
    for price in items:
        # Add all applicable coupons to the max-heap
        while coupon_index < M and coupons[coupon_index][0] <= price:
            heapq.heappush(max_heap, -coupons[coupon_index][1])  # Use negative for max-heap
            coupon_index += 1
        
        # Use the best discount available
        if max_heap:
            best_discount = -heapq.heappop(max_heap)
            total_cost += price - best_discount
        else:
            total_cost += price
    
    print(total_cost)