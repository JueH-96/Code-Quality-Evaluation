import heapq
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    K = int(input[ptr])
    ptr += 1
    
    # Build adjacency list
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        a = int(input[ptr])
        ptr += 1
        b = int(input[ptr])
        ptr += 1
        adj[a].append(b)
        adj[b].append(a)
    
    max_stamina = [-1] * (N + 1)
    heap = []
    
    # Initialize guards
    for _ in range(K):
        p = int(input[ptr])
        ptr += 1
        h = int(input[ptr])
        ptr += 1
        if max_stamina[p] < h:
            max_stamina[p] = h
            heapq.heappush(heap, (-h, p))
    
    # Process the heap
    while heap:
        s_neg, u = heapq.heappop(heap)
        s = -s_neg
        if s < max_stamina[u]:
            continue
        for v in adj[u]:
            new_s = s - 1
            if new_s > max_stamina[v]:
                max_stamina[v] = new_s
                heapq.heappush(heap, (-new_s, v))
    
    # Collect results
    result = []
    for v in range(1, N + 1):
        if max_stamina[v] >= 0:
            result.append(v)
    result.sort()
    print(len(result))
    print(' '.join(map(str, result)))

if __name__ == '__main__':
    main()