import sys
import heapq
from collections import deque

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    T = int(data[idx])
    idx += 1
    for _ in range(T):
        N = int(data[idx])
        M = int(data[idx+1])
        idx += 2
        boxes = []
        for _ in range(N):
            V = int(data[idx])
            P = int(data[idx+1])
            idx += 2
            boxes.append((P, V))
        boxes.sort()
        if M == 0:
            print(0)
            continue
        if N == 0:
            print(0)
            continue
        # Initialize the heap with M types, each with 0 capacity
        heap = [0] * M
        heapq.heapify(heap)
        total_cost = 0
        # Assign boxes to types
        for box in boxes:
            P, V = box
            # Assign the box to the type with smallest current capacity
            min_cap = heapq.heappop(heap)
            new_cap = min_cap + V
            heapq.heappush(heap, new_cap)
            total_cost += P
        # The minimal capacity across all types is the bottleneck
        min_cap = heap[0]
        # Net gain is min_cap - total_cost
        print(max(min_cap - total_cost, 0))

if __name__ == '__main__':
    main()