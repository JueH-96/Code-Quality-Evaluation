import sys
from collections import deque

def readints():
    return list(map(int, sys.stdin.readline().split()))

def find_cycle(perm, start):
    visited = [False] * (len(perm) + 1)
    cycle = []
    q = deque()
    q.append(start)
    visited[start] = True
    while q:
        u = q.popleft()
        cycle.append(u)
        v = perm[u-1]
        if not visited[v]:
            visited[v] = True
            q.append(v)
        else:
            break
    return cycle

def main():
    N, X = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    P = list(map(int, sys.stdin.readline().split()))
    Q = list(map(int, sys.stdin.readline().split()))
    
    # Find cycles in P and Q that include X
    cycle_P = find_cycle(P, X)
    cycle_Q = find_cycle(Q, X)
    
    # Check if all boxes with A[i] >0 are in cycle_P and all boxes with B[i] >0 are in cycle_Q
    valid = True
    for i in range(N):
        if i+1 == X:
            continue
        if A[i] > 0 and (i+1 not in cycle_P):
            valid = False
        if B[i] > 0 and (i+1 not in cycle_Q):
            valid = False
    if not valid:
        print(-1)
        return
    
    # Compute the intersection of cycle_P and cycle_Q
    set_P = set(cycle_P)
    set_Q = set(cycle_Q)
    intersection = set_P & set_Q
    
    len_P = len(cycle_P)
    len_Q = len(cycle_Q)
    
    # The answer is len_P + len_Q - len(intersection)
    print(len_P + len_Q - len(intersection))

if __name__ == "__main__":
    main()