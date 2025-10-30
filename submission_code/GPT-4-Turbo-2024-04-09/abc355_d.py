import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    intervals = []
    
    index = 1
    for _ in range(N):
        l = int(data[index])
        r = int(data[index + 1])
        intervals.append((l, r))
        index += 2
    
    # Sort intervals by the left endpoint
    intervals.sort()
    
    # To count the number of intersecting pairs
    intersect_count = 0
    
    # We will use a max heap (or priority queue) to keep track of the right endpoints
    import heapq
    min_heap = []
    
    for l, r in intervals:
        # Remove all intervals from the heap whose right endpoint is less than the current left endpoint
        while min_heap and min_heap[0] < l:
            heapq.heappop(min_heap)
        
        # All remaining intervals in the heap intersect with the current interval
        intersect_count += len(min_heap)
        
        # Add the current interval's right endpoint to the heap
        heapq.heappush(min_heap, r)
    
    print(intersect_count)

main()