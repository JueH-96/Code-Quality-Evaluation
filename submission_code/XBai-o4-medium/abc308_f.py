import heapq
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    n = int(input[ptr])
    ptr += 1
    m = int(input[ptr])
    ptr += 1
    
    p = list(map(int, input[ptr:ptr + n]))
    ptr += n
    
    l = list(map(int, input[ptr:ptr + m]))
    ptr += m
    
    d = list(map(int, input[ptr:ptr + m]))
    ptr += m
    
    coupons = list(zip(l, d))
    coupons.sort()
    
    p.sort()
    
    heap = []
    coupon_idx = 0
    total_discount = 0
    
    for price in p:
        while coupon_idx < m and coupons[coupon_idx][0] <= price:
            heapq.heappush(heap, -coupons[coupon_idx][1])
            coupon_idx += 1
        if heap:
            total_discount += -heapq.heappop(heap)
    
    sum_p = sum(p)
    print(sum_p - total_discount)

if __name__ == "__main__":
    main()