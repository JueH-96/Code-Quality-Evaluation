import sys
import heapq

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    from collections import defaultdict

    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    counts = defaultdict(int)
    idx = 1
    for _ in range(N):
        S = int(data[idx])
        C = int(data[idx+1])
        counts[S] += C
        idx += 2

    heap = list(counts.keys())
    heapq.heapify(heap)
    seen = set(heap)

    while heap:
        size = heapq.heappop(heap)
        c = counts[size]
        pairs = c // 2
        if pairs > 0:
            new_size = size * 2
            counts[new_size] += pairs
            if new_size not in seen:
                heapq.heappush(heap, new_size)
                seen.add(new_size)
            counts[size] %= 2

    total = sum(counts[size] for size in counts)
    print(total)

if __name__ == "__main__":
    main()