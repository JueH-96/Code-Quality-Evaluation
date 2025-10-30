import sys
import heapq

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    
    reverse_adj = [[] for _ in range(N+1)]  # 1-based indexing
    for _ in range(M):
        l = int(input[ptr])
        ptr += 1
        d = int(input[ptr])
        ptr += 1
        k = int(input[ptr])
        ptr += 1
        c = int(input[ptr])
        ptr += 1
        A = int(input[ptr])
        ptr += 1
        B = int(input[ptr])
        ptr += 1
        reverse_adj[B].append( (l, d, k, c, A) )
    
    INF = 10 ** 20
    latest = [ -INF for _ in range(N+1) ]
    latest[N] = INF
    heap = [ (-INF, N) ]
    heapq.heapify(heap)
    
    while heap:
        current_time_neg, u = heapq.heappop(heap)
        current_time = -current_time_neg
        if latest[u] > current_time:
            continue
        for edge in reverse_adj[u]:
            l_i, d_i, k_i, c_i, A_i = edge
            X = latest[u] - c_i
            if X < l_i:
                continue
            max_n = (X - l_i) // d_i
            n = min( max_n, k_i - 1 )
            if n < 0:
                continue
            t_candidate = l_i + n * d_i
            if t_candidate > latest[A_i]:
                latest[A_i] = t_candidate
                heapq.heappush(heap, (-t_candidate, A_i))
    
    for i in range(1, N):
        if latest[i] == -INF:
            print("Unreachable")
        else:
            print(latest[i])

if __name__ == "__main__":
    main()