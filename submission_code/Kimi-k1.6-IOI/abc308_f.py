import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    
    P = list(map(int, input[ptr:ptr+N]))
    ptr += N
    
    L = list(map(int, input[ptr:ptr+M]))
    ptr += M
    
    D = list(map(int, input[ptr:ptr+M]))
    ptr += M
    
    coupons = sorted(zip(L, D), key=lambda x: x[0])
    P_sorted = sorted(P)
    total_sum = sum(P_sorted)
    
    heap = []
    j = 0
    total_discount = 0
    
    for price in P_sorted:
        while j < M and coupons[j][0] <= price:
            heapq.heappush(heap, -coupons[j][1])
            j += 1
        if heap:
            total_discount += -heapq.heappop(heap)
    
    print(total_sum - total_discount)

if __name__ == "__main__":
    main()