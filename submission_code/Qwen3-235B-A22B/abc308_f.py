import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    P = list(map(int, data[idx:idx+N]))
    idx += N
    
    L = list(map(int, data[idx:idx+M]))
    idx += M
    
    D = list(map(int, data[idx:idx+M]))
    
    coupons = []
    for i in range(M):
        coupons.append((L[i], D[i]))
    
    # Sort coupons by L ascending
    coupons.sort(key=lambda x: x[0])
    
    # Sort items by price ascending
    sorted_P = sorted(P)
    
    sum_discount = 0
    i = 0  # Pointer for coupons
    heap = []
    
    for p in sorted_P:
        # Push all coupons with L <= p into the heap
        while i < M and coupons[i][0] <= p:
            # Push negative D for max heap
            heapq.heappush(heap, (-coupons[i][1], coupons[i][0]))
            i += 1
        
        if heap:
            best = heapq.heappop(heap)
            sum_discount += -best[0]
    
    total = sum(P) - sum_discount
    print(total)

if __name__ == "__main__":
    main()