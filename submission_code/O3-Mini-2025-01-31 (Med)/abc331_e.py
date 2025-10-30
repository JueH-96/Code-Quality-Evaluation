def main():
    import sys, heapq
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    l = int(next(it))
    
    a = [int(next(it)) for _ in range(n)]
    b = [int(next(it)) for _ in range(m)]
    
    # Build a dictionary mapping each main dish (0-indexed) to a set of side dishes (0-indexed) it cannot pair with.
    banned = {}
    for _ in range(l):
        c = int(next(it)) - 1
        d = int(next(it)) - 1
        if c not in banned:
            banned[c] = set()
        banned[c].add(d)

    # Sort main dishes and side dishes in descending order by their costs.
    # Each element is a tuple: (cost, original_index)
    sorted_mains = sorted([(a[i], i) for i in range(n)], key=lambda x: x[0], reverse=True)
    sorted_sides = sorted([(b[j], j) for j in range(m)], key=lambda x: x[0], reverse=True)

    # We use a max-heap to search candidate pairs in descending order of sum.
    # Our state will be indices (i, j) indicating the ith element from sorted_mains and jth element from sorted_sides.
    # We push negative sums to simulate a max-heap.
    heap = []
    visited = set()
    
    initial_sum = sorted_mains[0][0] + sorted_sides[0][0]
    heapq.heappush(heap, (-initial_sum, 0, 0))
    visited.add((0, 0))
    
    # Use a modification of the "K best sums" algorithm.
    while heap:
        neg_sum, i, j = heapq.heappop(heap)
        current_sum = -neg_sum
        # Convert sorted indices back to original dish indices.
        main_orig = sorted_mains[i][1]
        side_orig = sorted_sides[j][1]
        
        # Check if this combination is banned.
        if main_orig in banned and side_orig in banned[main_orig]:
            # If banned, add the neighboring candidate pairs:
            # Move one step down in main dishes or one step down in side dishes.
            if i + 1 < n and (i + 1, j) not in visited:
                next_sum = sorted_mains[i + 1][0] + sorted_sides[j][0]
                heapq.heappush(heap, (-next_sum, i + 1, j))
                visited.add((i + 1, j))
            if j + 1 < m and (i, j + 1) not in visited:
                next_sum = sorted_mains[i][0] + sorted_sides[j + 1][0]
                heapq.heappush(heap, (-next_sum, i, j + 1))
                visited.add((i, j + 1))
        else:
            # Found a valid pair: print and return.
            sys.stdout.write(str(current_sum))
            return

if __name__ == '__main__':
    main()