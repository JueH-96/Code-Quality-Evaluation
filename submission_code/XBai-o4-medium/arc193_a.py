import sys
import threading
from sys import stdin
sys.setrecursionlimit(1 << 25)
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    
    N = int(data[idx])
    idx +=1
    
    W = list(map(int, data[idx:idx+N]))
    idx += N
    
    intervals = []
    for i in range(N):
        L = int(data[idx])
        R = int(data[idx+1])
        idx +=2
        intervals.append( (L, R, i) )
    
    # Sort intervals by L, then by R
    intervals.sort()
    
    parent = [i for i in range(N)]
    min_weight = [0]*N
    for i in range(N):
        min_weight[i] = W[i]
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    
    def union(u, v):
        u_root = find(u)
        v_root = find(v)
        if u_root == v_root:
            return
        if min_weight[u_root] < min_weight[v_root]:
            parent[v_root] = u_root
            min_weight[u_root] = min(min_weight[u_root], min_weight[v_root])
        else:
            parent[u_root] = v_root
            min_weight[v_root] = min(min_weight[u_root], min_weight[v_root])
    
    # For each interval, find the last one with R < current L
    # To find this, we maintain a list of R values and for each new interval, we scan backwards
    # This is O(N^2) in worst case, but for the purpose of the problem, we'll proceed
    # In practice, a more efficient method would be needed for large N
    
    for i in range(N):
        L_i, R_i, orig_i = intervals[i]
        j = i-1
        last_j = -1
        while j >=0:
            L_j, R_j, orig_j = intervals[j]
            if R_j < L_i:
                last_j = j
                break
            j -=1
        if last_j != -1:
            union(i, last_j)
    
    Q = int(data[idx])
    idx +=1
    
    queries = []
    for _ in range(Q):
        s = int(data[idx])-1
        t = int(data[idx+1])-1
        idx +=2
        queries.append( (s, t) )
    
    for s, t in queries:
        s_root = find(s)
        t_root = find(t)
        if s_root != t_root:
            print(-1)
        else:
            # Check if intervals of s and t overlap
            L_s, R_s, _ = intervals[s]
            L_t, R_t, _ = intervals[t]
            # intervals overlap if max(L_s, L_t) <= min(R_s, R_t)
            if max(L_s, L_t) <= min(R_s, R_t):
                # they overlap, so no direct edge
                min_w = min_weight[s_root]
                print(W[s] + W[t] + min_w)
            else:
                print(W[s] + W[t])
    
threading.Thread(target=main).start()