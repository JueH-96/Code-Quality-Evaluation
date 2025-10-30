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
    
    max_stamina = [ -10**18 ] * (N + 1 )
    q = []

    for _ in range(K):
        p = int(input[idx]); idx +=1
        h = int(input[idx]); idx +=1
        if max_stamina[p] < h:
            max_stamina[p] = h
            heapq.heappush(q, (-h, p))
    
    while q:
        neg_s, u = heapq.heappop(q)
        s = -neg_s
        if s < max_stamina[u]:
            continue
        for v in adj[u]:
            new_s = s - 1
            if new_s > max_stamina[v]:
                max_stamina[v] = new_s
                heapq.heappush(q, (-new_s, v))
    
    res = []
    for v in range(1, N+1):
        if max_stamina[v] >= 0:
            res.append(v)
    res.sort()
    print(len(res))
    print(' '.join(map(str, res)))

if __name__ == '__main__':
    main()