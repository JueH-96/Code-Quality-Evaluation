import heapq

def main():
    import sys
    n, m = map(int, sys.stdin.readline().split())
    p = list(map(int, sys.stdin.readline().split()))
    l = list(map(int, sys.stdin.readline().split()))
    d = list(map(int, sys.stdin.readline().split()))
    
    coupons = list(zip(l, d))
    coupons.sort()  # Sort by L_i
    
    p_sorted = sorted(p)
    
    heap = []
    total = 0
    j = 0
    
    for price in p_sorted:
        while j < m and coupons[j][0] <= price:
            heapq.heappush(heap, -coupons[j][1])
            j += 1
        if heap:
            max_d = -heapq.heappop(heap)
            total += (price - max_d)
        else:
            total += price
    print(total)

if __name__ == "__main__":
    main()