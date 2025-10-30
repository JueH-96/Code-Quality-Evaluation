import sys
from collections import deque

def main():
    N, M = map(int, sys.stdin.readline().split())
    sets = []
    for _ in range(N):
        A = int(sys.stdin.readline())
        elements = list(map(int, sys.stdin.readline().split()))
        sets.append(elements)
    
    # Build sets_containing: for each element, list of set indices containing it
    sets_containing = [[] for _ in range(M + 1)]  # 1-based indexing
    for set_idx in range(N):
        for x in sets[set_idx]:
            sets_containing[x].append(set_idx)
    
    INF = float('inf')
    distance = [INF] * (M + 1)
    processed_sets = [False] * N
    q = deque()
    
    # Initialize with sets containing 1
    initial_sets = sets_containing[1]
    for set_idx in initial_sets:
        if not processed_sets[set_idx]:
            processed_sets[set_idx] = True
            for x in sets[set_idx]:
                if distance[x] > 1:
                    distance[x] = 1
                    q.append(x)
    
    # BFS
    while q:
        x = q.popleft()
        current_dist = distance[x]
        for set_idx in sets_containing[x]:
            if not processed_sets[set_idx]:
                processed_sets[set_idx] = True
                for y in sets[set_idx]:
                    if distance[y] > current_dist + 1:
                        distance[y] = current_dist + 1
                        q.append(y)
    
    if distance[M] == INF:
        print(-1)
    else:
        print(distance[M] - 1)

if __name__ == "__main__":
    main()