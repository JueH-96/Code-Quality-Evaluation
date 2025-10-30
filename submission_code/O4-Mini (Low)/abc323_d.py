import sys
import threading

def main():
    import sys
    import heapq
    
    input = sys.stdin.readline
    N = int(input().strip())
    
    # Read initial counts into a dict
    counts = {}
    for _ in range(N):
        s, c = map(int, input().split())
        counts[s] = counts.get(s, 0) + c
    
    # We'll process sizes in ascending order, merging counts on the fly.
    # Use a min‐heap of sizes to visit.
    heap = list(counts.keys())
    heapq.heapify(heap)
    in_heap = set(heap)
    
    while heap:
        s = heapq.heappop(heap)
        in_heap.remove(s)
        c = counts[s]
        # We can combine pairs of slimes of size s
        pairs = c // 2
        if pairs > 0:
            counts[s] = c % 2
            t = s * 2
            counts[t] = counts.get(t, 0) + pairs
            # If the new size hasn't been scheduled yet, push it
            if t not in in_heap:
                heapq.heappush(heap, t)
                in_heap.add(t)
    
    # After all possible merges, the total left is sum of counts (all are 0 or 1)
    result = sum(counts.values())
    print(result)

if __name__ == "__main__":
    main()