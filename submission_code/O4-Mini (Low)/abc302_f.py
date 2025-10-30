import sys
import threading
from collections import deque

def main():
    import sys
    data = sys.stdin
    N, M = map(int, data.readline().split())
    sets = []                  # list of lists: elements in each set
    elem_to_sets = [[] for _ in range(M+1)]
    start_sets = []            # indices of sets containing 1
    has_M = [False]*N          # flags for sets containing M
    # Read input
    zero_answer = False
    for i in range(N):
        A = int(data.readline().strip())
        arr = list(map(int, data.readline().split()))
        sets.append(arr)
        found1 = False
        foundM = False
        for v in arr:
            elem_to_sets[v].append(i)
            if v == 1:
                found1 = True
            if v == M:
                foundM = True
        if found1:
            start_sets.append(i)
        if foundM:
            has_M[i] = True
        # If a set has both 1 and M, answer is 0
        if found1 and foundM:
            print(0)
            return

    # If no set contains 1 or no set contains M, impossible
    if not start_sets or not any(has_M):
        print(-1)
        return

    # BFS on sets
    visited_set = [False]*N
    dist = [-1]*N
    processed_elem = [False]*(M+1)
    dq = deque()
    # initialize from all sets containing 1
    for si in start_sets:
        visited_set[si] = True
        dist[si] = 0
        dq.append(si)

    while dq:
        u = dq.popleft()
        d0 = dist[u]
        # Explore all elements of set u
        for v in sets[u]:
            if processed_elem[v]:
                continue
            processed_elem[v] = True
            # For each neighbor set through element v
            for w in elem_to_sets[v]:
                if not visited_set[w]:
                    visited_set[w] = True
                    dist[w] = d0 + 1
                    # If this set contains M, we are done
                    if has_M[w]:
                        print(dist[w])
                        return
                    dq.append(w)
    # If BFS completes without finding M
    print(-1)

if __name__ == "__main__":
    main()