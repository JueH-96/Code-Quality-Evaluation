import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    
    count = {}
    heap = []
    
    for _ in range(N):
        s = int(input[idx])
        c = int(input[idx + 1])
        idx += 2
        count[s] = c
        heapq.heappush(heap, s)
    
    while heap:
        s = heapq.heappop(heap)
        current = count[s]
        new = current // 2
        rem = current % 2
        count[s] = rem
        next_s = s * 2
        if next_s in count:
            count[next_s] += new
        else:
            count[next_s] = new
        if new > 0:
            heapq.heappush(heap, next_s)
    
    print(sum(count.values()))

if __name__ == "__main__":
    main()