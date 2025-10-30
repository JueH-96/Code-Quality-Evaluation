import sys
import heapq

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr +=1
    M = int(input[ptr]); ptr +=1
    K = int(input[ptr]); ptr +=1

    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        a = int(input[ptr]); ptr +=1
        b = int(input[ptr]); ptr +=1
        adj[a].append(b)
        adj[b].append(a)
    
    guards = []
    max_val = [-float('inf')] * (N+1)
    heap = []
    for _ in range(K):
        p = int(input[ptr]); ptr +=1
        h = int(input[ptr]); ptr +=1
        max_val[p] = h
        heapq.heappush(heap, (-h, p))
    
    while heap:
        current_neg_val, u = heapq.heappop(heap)
        current_val = -current_neg_val
        if current_val < max_val[u]:
            continue
        for v in adj[u]:
            new_val = current_val -1
            if new_val > max_val[v]:
                max_val[v] = new_val
                heapq.heappush(heap, (-new_val, v))
    
    result = [v for v in range(1, N+1) if max_val[v] >=0]
    result.sort()
    print(len(result))
    print(' '.join(map(str, result)))

if __name__ == '__main__':
    main()