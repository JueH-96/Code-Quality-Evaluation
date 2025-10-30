import heapq

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    prices = list(map(int, data[2:2+n]))
    Ls = list(map(int, data[2+n:2+n+m]))
    Ds = list(map(int, data[2+n+m:2+n+2*m]))
    
    total_regular = sum(prices)
    prices.sort()
    coupons = []
    for i in range(m):
        coupons.append((Ls[i], Ds[i]))
    coupons.sort(key=lambda x: x[0])
    
    heap = []
    j = 0
    total_discount = 0
    
    for p in prices:
        while j < m and coupons[j][0] <= p:
            heapq.heappush(heap, -coupons[j][1])
            j += 1
        if heap:
            total_discount += -heapq.heappop(heap)
            
    result = total_regular - total_discount
    print(result)

if __name__ == "__main__":
    main()