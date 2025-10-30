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
    
    infos = [[] for _ in range(N+1)]  # infos[B] contains tuples (A, l, d, k, c)
    
    for _ in range(M):
        l = int(data[idx])
        d = int(data[idx+1])
        k = int(data[idx+2])
        c = int(data[idx+3])
        A = int(data[idx+4])
        B = int(data[idx+5])
        idx +=6
        infos[B].append( (A, l, d, k, c) )
    
    D = [ -float('inf') ] * (N+1)
    D[N] = float('inf')
    
    heap = []
    heapq.heappush(heap, (-D[N], N))
    
    while heap:
        current_dist_neg, u = heapq.heappop(heap)
        current_dist = -current_dist_neg
        
        if current_dist < D[u]:
            continue
        
        for info in infos[u]:
            A, l, d, k, c = info
            X = current_dist - c
            
            if X < l:
                continue
            
            m = (X - l) // d
            if m < 0:
                continue
            m = min(m, k-1)
            t = l + m * d
            if t > D[A]:
                D[A] = t
                heapq.heappush(heap, (-t, A))
    
    for i in range(1, N):
        if D[i] == -float('inf'):
            print("Unreachable")
        else:
            print(D[i])

if __name__ == '__main__':
    main()