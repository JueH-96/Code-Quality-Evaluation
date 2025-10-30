import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    sets = []
    has_1_and_M = False
    for _ in range(N):
        A = int(sys.stdin.readline())
        s = list(map(int, sys.stdin.readline().split()))
        sets.append(s)
        if 1 in s and M in s:
            has_1_and_M = True
    if has_1_and_M:
        print(0)
        return
    
    # DSU to check if 1 and M are connected
    parent = list(range(M + 1))
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    def union(u, v):
        u_root = find(u)
        v_root = find(v)
        if u_root != v_root:
            parent[v_root] = u_root
    for s in sets:
        if not s:
            continue
        first = s[0]
        for elem in s[1:]:
            union(first, elem)
    if find(1) != find(M):
        print(-1)
        return
    
    # Preprocess element to sets mapping
    elem_to_sets = [[] for _ in range(M + 1)]
    set_contains_1 = []
    set_contains_M = []
    for i, s in enumerate(sets):
        elem_set = set(s)
        for elem in elem_set:
            elem_to_sets[elem].append(i)
        has_1 = 1 in elem_set
        has_M = M in elem_set
        set_contains_1.append(has_1)
        set_contains_M.append(has_M)
    
    # BFS on sets
    visited = [False] * N
    q = deque()
    for i in range(N):
        if set_contains_1[i]:
            visited[i] = True
            q.append(i)
    # Check if any initial set contains M
    for i in range(N):
        if visited[i] and set_contains_M[i]:
            print(0)
            return
    distance = 0
    while q:
        distance += 1
        for _ in range(len(q)):
            u = q.popleft()
            current_set = sets[u]
            for elem in current_set:
                for v in elem_to_sets[elem]:
                    if not visited[v]:
                        if set_contains_M[v]:
                            print(distance)
                            return
                        visited[v] = True
                        q.append(v)
    print(-1)

if __name__ == '__main__':
    main()