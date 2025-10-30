import sys
from collections import deque

def main():
    n, m = map(int, sys.stdin.readline().split())
    element_adj = [[] for _ in range(m + 1)]  # element 0 unused
    set_adj = []
    for _ in range(n):
        a = int(sys.stdin.readline())
        s = list(map(int, sys.stdin.readline().split()))
        set_adj.append(s)
        for e in s:
            element_adj[e].append(len(set_adj) - 1)
    
    ele_dist = [-1] * (m + 1)
    set_dist = [-1] * n
    q = deque()
    ele_dist[1] = 0
    q.append((0, 1))  # (type, id): 0 for element, 1 for set
    
    while q:
        typ, idx = q.popleft()
        if typ == 0:  # element node
            for s in element_adj[idx]:
                if set_dist[s] == -1:
                    set_dist[s] = ele_dist[idx] + 1
                    q.append((1, s))
        else:  # set node
            for e in set_adj[idx]:
                if ele_dist[e] == -1:
                    ele_dist[e] = set_dist[idx] + 1
                    q.append((0, e))
    
    if ele_dist[m] == -1:
        print(-1)
    else:
        print((ele_dist[m] // 2) - 1)

if __name__ == "__main__":
    main()