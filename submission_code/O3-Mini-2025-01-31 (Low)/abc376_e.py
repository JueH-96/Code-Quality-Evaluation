def main():
    import sys, heapq
    data = sys.stdin.read().split()
    t = int(data[0])
    pos = 1
    results = []
    for _ in range(t):
        n = int(data[pos])
        k = int(data[pos+1])
        pos += 2
        A = list(map(int, data[pos: pos+n]))
        pos += n
        B = list(map(int, data[pos: pos+n]))
        pos += n
        
        # Pair up A and B so we can sort by A.
        pairs = list(zip(A, B))
        pairs.sort(key=lambda x: x[0])
        
        # We use a max-heap to keep track of k smallest B's among the already processed items.
        # Python's heapq is a min-heap; so we insert negative B to simulate a max-heap.
        heap = []
        b_sum = 0
        best = float('inf')
        
        for a, b in pairs:
            b_sum += b
            heapq.heappush(heap, -b)  # Push negative so largest b is at the top.
            if len(heap) > k:
                # Remove the largest b from the chosen set (i.e. smallest negative becomes removed).
                b_removed = -heapq.heappop(heap)
                b_sum -= b_removed
            if len(heap) == k:
                # current a is the maximum because sorted by A.
                product = a * b_sum
                if product < best:
                    best = product
        results.append(str(best))
    
    sys.stdout.write("
".join(results))
    
if __name__ == '__main__':
    main()