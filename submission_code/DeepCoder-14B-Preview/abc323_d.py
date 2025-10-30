import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    n = int(input[idx])
    idx += 1
    freq = {}
    for _ in range(n):
        s = int(input[idx])
        c = int(input[idx + 1])
        idx += 2
        if s in freq:
            freq[s] += c
        else:
            freq[s] = c
    # Create a min-heap
    heap = list(freq.keys())
    heapq.heapify(heap)
    # Process the heap
    while heap:
        x = heapq.heappop(heap)
        if x not in freq or freq[x] < 1:
            continue
        # Calculate pairs and rem
        pairs = freq[x] // 2
        rem = freq[x] % 2
        freq[x] = rem
        new_slimes = pairs
        if new_slimes > 0:
            if 2 * x in freq:
                freq[2 * x] += new_slimes
            else:
                freq[2 * x] = new_slimes
            heapq.heappush(heap, 2 * x)
    # Sum all the values in freq
    total = sum(freq.values())
    print(total)

if __name__ == '__main__':
    main()