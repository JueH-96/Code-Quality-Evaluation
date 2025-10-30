def main():
    import sys
    import heapq
    data = sys.stdin.read().strip().split()
    if not data:
        return
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        n = int(data[index])
        k = int(data[index+1])
        index += 2
        A = list(map(int, data[index:index+n]))
        index += n
        B = list(map(int, data[index:index+n]))
        index += n
        
        # Create pairs (A_i, B_i) and sort by A (non-decreasing)
        pairs = list(zip(A, B))
        pairs.sort(key=lambda x: x[0])
        
        # Use a max-heap to keep track of the K smallest B_i's seen so far.
        # We'll store negative values because Python's heapq implements a min-heap.
        curr_heap = []
        sumb = 0
        ans = 10**18  # a sufficiently large number
        for a, b in pairs:
            heapq.heappush(curr_heap, -b)
            sumb += b
            if len(curr_heap) > k:
                # Remove the index that has the highest B value 
                removed = -heapq.heappop(curr_heap)
                sumb -= removed
            if len(curr_heap) == k:
                # In the current subset the maximum A is at most a
                candidate = a * sumb
                if candidate < ans:
                    ans = candidate
        results.append(str(ans))
    sys.stdout.write("
".join(results))

if __name__ == '__main__':
    main()