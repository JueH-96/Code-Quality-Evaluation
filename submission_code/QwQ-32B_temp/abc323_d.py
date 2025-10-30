import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    
    counts = {}
    heap = []
    
    for _ in range(N):
        s = int(input[idx])
        c = int(input[idx+1])
        idx += 2
        counts[s] = c
        heapq.heappush(heap, s)
    
    while heap:
        s = heapq.heappop(heap)
        current = counts.get(s, 0)
        remainder = current % 2
        pairs = current // 2
        
        counts[s] = remainder
        if pairs > 0:
            next_s = 2 * s
            counts[next_s] = counts.get(next_s, 0) + pairs
            heapq.heappush(heap, next_s)
    
    total = sum(counts.values())
    print(total)

if __name__ == "__main__":
    main()