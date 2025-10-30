import heapq
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    index = 1
    slimes = []
    
    for _ in range(n):
        s = int(data[index])
        c = int(data[index + 1])
        slimes.append((s, c))
        index += 2
    
    counts = defaultdict(int)
    for s, c in slimes:
        counts[s] = c
    
    heap = []
    for s in counts:
        heapq.heappush(heap, s)
    
    total = 0
    
    while heap:
        s = heapq.heappop(heap)
        current_c = counts[s]
        if current_c <= 0:
            continue
        total += current_c % 2
        pairs = current_c // 2
        new_s = s * 2
        counts[new_s] += pairs
        heapq.heappush(heap, new_s)
    
    print(total)

if __name__ == '__main__':
    main()