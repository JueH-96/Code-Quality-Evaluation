# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    INF = float('inf')
    f = [-float('inf')] * (N + 1)
    f[N] = INF
    in_queue = [False] * (N + 1)
    from collections import deque

    # Build reverse adjacency list: For each station B_i, store the train patterns that arrive at B_i
    adj = [[] for _ in range(N +1)]  # adj[B_i] = list of train patterns that arrive at B_i

    for _ in range(M):
        l_i, d_i, k_i, c_i, A_i, B_i = map(int, sys.stdin.readline().split())
        adj[B_i].append( (l_i, d_i, k_i, c_i, A_i, B_i) )

    queue = deque()
    queue.append(N)
    in_queue[N] = True

    while queue:
        S = queue.popleft()
        in_queue[S] = False
        t_arr_max = f[S]
        # For all train patterns that can reach S
        for pattern in adj[S]:
            l_i, d_i, k_i, c_i, A_i, B_i = pattern
            t_dep_max = t_arr_max - c_i
            if t_dep_max < l_i:
                continue  # No valid departure
            j_max = (t_dep_max - l_i) // d_i
            if j_max < 0:
                continue
            j_max = min(j_max, k_i -1)
            t_dep = l_i + j_max * d_i
            if f[A_i] < t_dep:
                f[A_i] = t_dep
                if not in_queue[A_i]:
                    queue.append(A_i)
                    in_queue[A_i] = True

    for S in range(1, N):
        if f[S] == -float('inf'):
            print('Unreachable')
        else:
            # Since times could be large, print as int
            print(int(f[S]))
    

if __name__ == '__main__':
    threading.Thread(target=main).start()