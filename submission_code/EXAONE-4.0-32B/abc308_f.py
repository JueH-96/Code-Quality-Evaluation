import heapq
import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    P = list(map(int, data[2:2+n]))
    L_list = list(map(int, data[2+n:2+n+m]))
    D_list = list(map(int, data[2+n+m:2+n+2*m]))
    
    coupons = []
    for i in range(m):
        coupons.append((L_list[i], D_list[i]))
    
    P.sort()
    coupons.sort(key=lambda x: x[0])
    
    heap = []
    total_regular = sum(P)
    j = 0
    total_discount = 0
    
    for i in range(n):
        while j < m and coupons[j][0] <= P[i]:
            heapq.heappush(heap, -coupons[j][1])
            j += 1
        if heap:
            max_discount = -heapq.heappop(heap)
            total_discount += max_discount
            
    result = total_regular - total_discount
    print(result)

if __name__ == "__main__":
    main()