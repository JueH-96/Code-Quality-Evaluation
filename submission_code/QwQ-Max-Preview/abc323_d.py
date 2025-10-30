import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    n = int(input[idx])
    idx += 1
    
    counts = {}
    heap = []
    processed = set()
    
    for _ in range(n):
        s = int(input[idx])
        c = int(input[idx+1])
        idx += 2
        counts[s] = c
        heapq.heappush(heap, s)
    
    while heap:
        x = heapq.heappop(heap)
        if x in processed:
            continue
        processed.add(x)
        current = counts.get(x, 0)
        merges = current // 2
        remaining = current % 2
        counts[x] = remaining
        if merges > 0:
            next_size = x * 2
            counts[next_size] = counts.get(next_size, 0) + merges
            if next_size not in processed:
                heapq.heappush(heap, next_size)
    
    total = sum(counts.values())
    print(total)

if __name__ == "__main__":
    main()