import heapq
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    index = 1
    count = defaultdict(int)
    for _ in range(n):
        s = int(data[index])
        c = int(data[index + 1])
        index += 2
        count[s] = c
    
    heap = list(count.keys())
    heapq.heapify(heap)
    in_heap = set(heap)
    
    while heap:
        x = heapq.heappop(heap)
        if x not in in_heap:
            continue
        in_heap.remove(x)
        
        current = count[x]
        rem = current % 2
        add = current // 2
        count[x] = rem
        
        if add > 0:
            next_x = 2 * x
            count[next_x] += add
            if next_x not in in_heap:
                heapq.heappush(heap, next_x)
                in_heap.add(next_x)
    
    total = sum(count.values())
    print(total)

if __name__ == "__main__":
    main()