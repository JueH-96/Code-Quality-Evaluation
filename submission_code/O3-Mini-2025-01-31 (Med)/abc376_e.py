def main():
    import sys, heapq
    data = sys.stdin.read().split()
    if not data:
        return
    t = int(data[0])
    pos = 1
    outputs = []
    # Process each test case
    for _ in range(t):
        n = int(data[pos]); k = int(data[pos+1]); pos += 2
        A = list(map(int, data[pos:pos+n])); pos += n
        B = list(map(int, data[pos:pos+n])); pos += n

        # Combine A and B and sort by A in ascending order.
        AB = list(zip(A, B))
        AB.sort(key=lambda x: x[0])

        # We'll maintain a max-heap for B's (simulate via negative values) and a running sum.
        heap = []
        best = None
        b_sum = 0

        # For each element in sorted order, treat the current element's A as the candidate maximum.
        for a, b in AB:
            heapq.heappush(heap, -b)
            b_sum += b

            # Remove extra element if we have more than k elements so that we are
            # always keeping the K smallest B values from the ones we've processed.
            if len(heap) > k:
                removed = -heapq.heappop(heap)
                b_sum -= removed

            # Once we have exactly k elements, calculate the value.
            if len(heap) == k:
                current_value = a * b_sum
                if best is None or current_value < best:
                    best = current_value

        outputs.append(str(best))
    sys.stdout.write("
".join(outputs))

if __name__ == '__main__':
    main()