# YOUR CODE HERE
import sys
import threading
import heapq

def main():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    incoming_trains = [[] for _ in range(N+1)]
    for _ in range(M):
        parts = sys.stdin.readline().split()
        if len(parts) != 6:
            # Handle possible incorrect lines
            continue
        l, d, k, c, A, B = map(int, parts)
        incoming_trains[B].append( (l, d, k, c, A) )
    
    f = [-1] * (N+1)
    f[N] = 10**20
    heap = []
    heapq.heappush(heap, (-f[N], N))
    
    while heap:
        neg_current_f, S = heapq.heappop(heap)
        current_f = -neg_current_f
        if current_f < f[S]:
            continue
        for train in incoming_trains[S]:
            l_i, d_i, k_i, c_i, A_i = train
            T = f[S] - c_i
            if T < l_i:
                continue
            x_max = (T - l_i) // d_i
            if x_max <0:
                continue
            if x_max >=k_i:
                x_max = k_i -1
            t = l_i + d_i *x_max
            if t > f[A_i]:
                f[A_i] = t
                heapq.heappush(heap, (-t, A_i))
    
    for S in range(1, N):
        if f[S] >=0 and f[S] <10**20:
            print(f[S])
        elif f[S] >=10**20:
            print(10**20)
        else:
            print("Unreachable")

threading.Thread(target=main).start()