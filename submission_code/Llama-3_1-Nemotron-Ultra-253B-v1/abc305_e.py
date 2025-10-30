import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx +=1
    M = int(data[idx])
    idx +=1
    K = int(data[idx])
    idx +=1
    
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        a = int(data[idx])
        idx +=1
        b = int(data[idx])
        idx +=1
        adj[a].append(b)
        adj[b].append(a)
    
    max_values = [-float('inf')] * (N + 1)
    heap = []
    
    for _ in range(K):
        p = int(data[idx])
        idx +=1
        h = int(data[idx])
        idx +=1
        if h > max_values[p]:
            max_values[p] = h
            heapq.heappush(heap, (-h, p))
    
    while heap:
        current_neg_val, u = heapq.heappop(heap)
        current_val = -current_neg_val
        if current_val < max_values[u]:
            continue
        for v in adj[u]:
            new_val = current_val - 1
            if new_val > max_values[v]:
                max_values[v] = new_val
                heapq.heappush(heap, (-new_val, v))
    
    result = []
    for i in range(1, N+1):
        if max_values[i] >= 0:
            result.append(i)
    result.sort()
    print(len(result))
    print(' '.join(map(str, result)))

if __name__ == '__main__':
    main()