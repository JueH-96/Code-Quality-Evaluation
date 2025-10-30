import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    P = list(map(int, data[2:2+N]))
    L = list(map(int, data[2+N:2+N+M]))
    D = list(map(int, data[2+N+M:2+N+2*M]))
    
    # Sort items in descending order of P_i
    items = sorted(P, reverse=True)
    
    # Sort coupons by L_i ascending
    coupons = sorted(zip(L, D), key=lambda x: x[0])
    
    total_cost = sum(P)
    total_discount = 0
    
    # Initialize a max heap for D_i
    heap = []
    coupon_ptr = 0
    
    for price in items:
        # Add all coupons with L_i <= price to the heap
        while coupon_ptr < M and coupons[coupon_ptr][0] <= price:
            # Use negative D_i for max heap
            heapq.heappush(heap, -coupons[coupon_ptr][1])
            coupon_ptr += 1
        if heap:
            # Assign the coupon with the highest D_i
            max_d = -heapq.heappop(heap)
            total_discount += max_d
    
    # Calculate the minimum total cost
    min_total_cost = total_cost - total_discount
    print(min_total_cost)

if __name__ == '__main__':
    main()