import sys
import heapq

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    idx += 2
    P = list(map(int, data[idx:idx+N]))
    idx += N
    L = list(map(int, data[idx:idx+M]))
    idx += M
    D = list(map(int, data[idx:idx+M]))
    
    # Pair L and D together and sort based on L
    coupons = list(zip(L, D))
    coupons.sort()
    
    # Sort items
    P.sort()
    
    total = 0
    j = 0
    heap = []
    
    for p in P:
        # Add all coupons where L_i <= p to the heap
        while j < M and coupons[j][0] <= p:
            # Use a max-heap by pushing negative D_i
            heapq.heappush(heap, -coupons[j][1])
            j += 1
        if heap:
            # Use the coupon with the largest D_i
            discount = -heapq.heappop(heap)
            total += max(0, p - discount)
        else:
            total += p
    
    print(total)

if __name__ == "__main__":
    main()