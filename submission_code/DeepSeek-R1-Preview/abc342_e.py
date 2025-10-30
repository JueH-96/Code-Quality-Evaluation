import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    adj = [[] for _ in range(N + 1)]  # 1-based indexing

    for _ in range(M):
        l = int(data[idx])
        idx +=1
        d = int(data[idx])
        idx +=1
        k = int(data[idx])
        idx +=1
        c = int(data[idx])
        idx +=1
        A = int(data[idx])
        idx +=1
        B = int(data[idx])
        idx +=1
        adj[B].append( (l, d, k, c, A) )
    
    L = [ -float('inf') ] * (N + 1)
    L[N] = 10**18  # Treat N's L as a very high value

    heap = []
    heapq.heappush(heap, (-L[N], N))

    while heap:
        neg_current_L, B = heapq.heappop(heap)
        current_L = -neg_current_L

        if current_L < L[B]:
            continue

        for edge in adj[B]:
            l_i, d_i, k_i, c_i, A_i = edge

            if B == N:
                # Special case: arriving at N, so t is the last possible in the schedule
                t = l_i + (k_i - 1) * d_i
            else:
                X = current_L - c_i
                if X < l_i:
                    continue
                m = (X - l_i) // d_i
                m = min(m, k_i - 1)
                if m < 0:
                    continue
                t = l_i + m * d_i

            if t > L[A_i]:
                L[A_i] = t
                heapq.heappush(heap, (-t, A_i))
    
    for i in range(1, N):
        if L[i] == -float('inf'):
            print("Unreachable")
        else:
            print(L[i])

if __name__ == '__main__':
    main()