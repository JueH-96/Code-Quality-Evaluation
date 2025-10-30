import sys
import heapq

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx +=1
    M = int(input[idx]); idx +=1
    K = int(input[idx]); idx +=1

    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        a = int(input[idx]); idx +=1
        b = int(input[idx]); idx +=1
        adj[a].append(b)
        adj[b].append(a)
    
    max_remaining = [-1] * (N + 1)  # Using -1 as a placeholder for -infinity since distances can't be negative
    guards = []
    for _ in range(K):
        p = int(input[idx]); idx +=1
        h = int(input[idx]); idx +=1
        max_remaining[p] = h  # p_i are distinct, so no need to check max
        guards.append((p, h))
    
    # Using a max-heap implemented with negative values
    heap = []
    for p, h in guards:
        heapq.heappush(heap, (-h, p))
    
    while heap:
        current_neg, u = heapq.heappop(heap)
        current = -current_neg
        
        # Skip if a better remaining has already been processed
        if current < max_remaining[u]:
            continue
        
        for v in adj[u]:
            new_remaining = current - 1
            if new_remaining > max_remaining[v]:
                max_remaining[v] = new_remaining
                heapq.heappush(heap, (-new_remaining, v))
    
    # Collect guarded vertices
    result = []
    for v in range(1, N+1):
        if max_remaining[v] >= 0:
            result.append(v)
    
    result.sort()
    print(len(result))
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()