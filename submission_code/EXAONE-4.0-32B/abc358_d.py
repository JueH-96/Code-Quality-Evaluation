import heapq
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    
    n = int(data[0])
    m = int(data[1])
    A = list(map(int, data[2:2+n]))
    B = list(map(int, data[2+n:2+n+m]))
    
    A.sort()
    B.sort(reverse=True)
    
    heap = []
    total_cost = 0
    j = n - 1
    
    for b in B:
        while j >= 0 and A[j] >= b:
            heapq.heappush(heap, A[j])
            j -= 1
        
        if not heap:
            print(-1)
            return
        
        box = heapq.heappop(heap)
        total_cost += box
        
    print(total_cost)

if __name__ == "__main__":
    main()