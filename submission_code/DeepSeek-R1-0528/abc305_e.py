import sys
import heapq

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
        
    it = iter(data)
    n = int(next(it)); m = int(next(it)); k = int(next(it))
    
    graph = [[] for _ in range(n+1)]
    for i in range(m):
        a = int(next(it)); b = int(next(it))
        graph[a].append(b)
        graph[b].append(a)
        
    guards = []
    rem = [-10**18] * (n+1)
    heap = []
    
    for i in range(k):
        p = int(next(it)); h = int(next(it))
        guards.append((p, h))
        rem[p] = h
        heapq.heappush(heap, (-h, p))
        
    while heap:
        neg_rem, u = heapq.heappop(heap)
        current_rem = rem[u]
        if -neg_rem != current_rem:
            continue
        for v in graph[u]:
            candidate = current_rem - 1
            if candidate > rem[v]:
                rem[v] = candidate
                heapq.heappush(heap, (-candidate, v))
                
    guarded = []
    for i in range(1, n+1):
        if rem[i] >= 0:
            guarded.append(i)
            
    guarded.sort()
    print(len(guarded))
    if guarded:
        print(" ".join(map(str, guarded)))
    else:
        print()

if __name__ == "__main__":
    main()