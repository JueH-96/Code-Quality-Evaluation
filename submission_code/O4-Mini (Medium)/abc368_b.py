import sys
import heapq

def main():
    input = sys.stdin.readline
    N = int(input().strip())
    A = list(map(int, input().split()))
    
    # Build a max-heap using negatives
    heap = []
    for a in A:
        if a > 0:
            heapq.heappush(heap, -a)
    
    ops = 0
    # While there are at least two positive elements
    while len(heap) >= 2:
        # Pop two largest
        a = -heapq.heappop(heap)
        b = -heapq.heappop(heap)
        
        # Perform the operation: decrease both by 1
        a -= 1
        b -= 1
        ops += 1
        
        # If still positive, push back
        if a > 0:
            heapq.heappush(heap, -a)
        if b > 0:
            heapq.heappush(heap, -b)
    
    print(ops)

if __name__ == "__main__":
    main()