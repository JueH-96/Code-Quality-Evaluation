# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, Q = map(int, sys.stdin.readline().split())
    cnt_zero_deg = N
    deg = [0] * (N + 1)
    adj = [set() for _ in range(N + 1)]

    for _ in range(Q):
        query = sys.stdin.readline().split()
        if query[0] == '1':
            u = int(query[1])
            v = int(query[2])
            adj[u].add(v)
            adj[v].add(u)
            if deg[u]==0:
                cnt_zero_deg -=1
            deg[u]+=1
            if deg[v]==0:
                cnt_zero_deg -=1
            deg[v]+=1
        elif query[0] == '2':
            v = int(query[1])
            if deg[v]>0:
                for u in adj[v]:
                    adj[u].discard(v)
                    deg[u]-=1
                    if deg[u]==0:
                        cnt_zero_deg +=1
                adj[v].clear()
                deg[v]=0
                cnt_zero_deg +=1  # since deg[v] becomes zero now
        print(cnt_zero_deg)
threading.Thread(target=main).start()