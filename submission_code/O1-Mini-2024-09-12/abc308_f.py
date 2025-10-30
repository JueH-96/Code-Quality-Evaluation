# YOUR CODE HERE
import sys
import heapq

def main():
    import sys
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx]); idx +=1
    M = int(data[idx]); idx +=1
    P = []
    for _ in range(N):
        P.append(int(data[idx]))
        idx +=1
    L = []
    for _ in range(M):
        L.append(int(data[idx]))
        idx +=1
    D = []
    for _ in range(M):
        D.append(int(data[idx]))
        idx +=1
    items = sorted(P)
    coupons = sorted(zip(L,D), key=lambda x: x[0])
    heap = []
    total_discount = 0
    j =0
    for price in items:
        while j < M and coupons[j][0] <= price:
            heapq.heappush(heap, -coupons[j][1])
            j +=1
        if heap:
            total_discount += -heapq.heappop(heap)
    total_cost = sum(P) - total_discount
    print(total_cost)

if __name__ == "__main__":
    main()