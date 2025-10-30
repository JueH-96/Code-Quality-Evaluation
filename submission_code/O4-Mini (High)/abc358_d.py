import sys
import threading

def main():
    import sys
    data = sys.stdin.readline
    N, M = map(int, data().split())
    A = list(map(int, data().split()))
    B = list(map(int, data().split()))
    # Sort boxes by candy/price ascending, demands descending
    A.sort()
    B.sort(reverse=True)
    import heapq
    heap = []
    push = heapq.heappush
    pop = heapq.heappop

    i = N - 1
    total = 0
    for b in B:
        # Push all boxes that can satisfy demand b
        while i >= 0 and A[i] >= b:
            push(heap, A[i])
            i -= 1
        # If no available box, impossible
        if not heap:
            print(-1)
            return
        # Use the cheapest eligible box
        total += pop(heap)
    print(total)

if __name__ == "__main__":
    main()