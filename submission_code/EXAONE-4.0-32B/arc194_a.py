import heapq
import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    a = list(map(int, data[1:1+n]))
    
    if n == 1:
        print(a[0])
        return
        
    heap = []
    total = 0
    for i in range(n):
        heapq.heappush(heap, a[i])
        total += a[i]
        if len(heap) > n - i - 1:
            total -= heapq.heappop(heap)
            
    print(total)

if __name__ == "__main__":
    main()