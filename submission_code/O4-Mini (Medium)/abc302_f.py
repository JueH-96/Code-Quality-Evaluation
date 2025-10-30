import sys
import threading

def main():
    import sys
    from collections import deque

    input = sys.stdin.readline
    N_M = input().split()
    if not N_M:
        print(-1)
        return
    N, M = map(int, N_M)
    N = int(N)
    M = int(M)

    # element_to_sets[e] = list of set-indices i that contain element e
    element_to_sets = [[] for _ in range(M+1)]
    sets = []  # sets[i] = list of elements in set i
    contains_M = [False]*N  # mark which sets contain M

    for i in range(N):
        line = input().strip()
        while line == "":
            line = input().strip()
        A_i = int(line)
        elems = list(map(int, input().split()))
        sets.append(elems)
        has1 = False
        hasM = False
        for e in elems:
            element_to_sets[e].append(i)
            if e == 1:
                has1 = True
            if e == M:
                hasM = True
        contains_M[i] = hasM

    # If any set already contains both 1 and M, answer is 0
    for i in range(N):
        if contains_M[i] and (1 in sets[i]):
            print(0)
            return

    # Prepare BFS from all sets that contain 1
    dist = [-1]*N
    vis_set = [False]*N
    vis_elem = [False]*(M+1)
    q = deque()

    # Initialize queue with sets containing 1
    for i in range(N):
        # We didn't store contains_1 array; test 1 in sets[i]
        # But scanning sets[i] each time is O(A_i). Instead, we can track at input.
        # Simpler: let's do it now.
        pass

    # Let's re-scan to find starting sets
    for i in range(N):
        # Since sum of A_i is manageable, we can check membership directly
        # But to optimize, we could have kept a contains_1 list; given constraints it's fine.
        if 1 in sets[i]:
            dist[i] = 0
            vis_set[i] = True
            q.append(i)

    # If no starting sets, impossible
    if not q:
        print(-1)
        return

    # BFS
    while q:
        i = q.popleft()
        # If this set contains M, we found a path
        if contains_M[i]:
            print(dist[i])
            return
        # For each element in sets[i], traverse to other sets via that element
        for e in sets[i]:
            if not vis_elem[e]:
                vis_elem[e] = True
                # Traverse all sets that contain e
                for j in element_to_sets[e]:
                    if not vis_set[j]:
                        vis_set[j] = True
                        dist[j] = dist[i] + 1
                        q.append(j)
                # Clear list to save memory / avoid re-traversal
                element_to_sets[e].clear()

    # If BFS completes without finding a set with M
    print(-1)

if __name__ == "__main__":
    main()