def main():
    import sys
    import heapq

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    freq = {}

    # Read input and aggregate slime counts by size
    idx = 1
    for _ in range(N):
        S = int(input_data[idx]); C = int(input_data[idx+1])
        idx += 2
        freq[S] = freq.get(S, 0) + C

    # Build a min-heap of (size, count)
    heap = [(sz, cnt) for sz, cnt in freq.items()]
    heapq.heapify(heap)

    # Process merges in bulk via a lazy (size, count) pop
    while heap:
        size, count_on_heap = heapq.heappop(heap)
        # If this entry is outdated, skip it
        if freq[size] != count_on_heap:
            continue
        # If we have at least 2 slimes of this size, merge them
        if count_on_heap >= 2:
            merged = count_on_heap // 2
            leftover = count_on_heap % 2
            freq[size] = leftover
            new_size = size * 2
            freq[new_size] = freq.get(new_size, 0) + merged
            # Push updated new_size count back to the heap
            heapq.heappush(heap, (new_size, freq[new_size]))

    # The answer is the total number of slimes remaining
    print(sum(freq.values()))

# Don't forget to call main()
if __name__ == "__main__":
    main()