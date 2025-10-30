import sys
import threading

def main():
    import sys
    import heapq

    input = sys.stdin.readline
    N = int(input().strip())
    A = list(map(int, input().split()))
    
    # Use a max-heap: Python has a min-heap, so we store negatives.
    heap = []
    for a in A:
        if a > 0:
            heapq.heappush(heap, -a)

    ops = 0
    # While there are at least two positive elements left
    while len(heap) >= 2:
        # Pop two largest elements
        x = -heapq.heappop(heap)
        y = -heapq.heappop(heap)
        # Decrement both
        x -= 1
        y -= 1
        ops += 1
        # Push back if still positive
        if x > 0:
            heapq.heappush(heap, -x)
        if y > 0:
            heapq.heappush(heap, -y)

    print(ops)

if __name__ == "__main__":
    main()