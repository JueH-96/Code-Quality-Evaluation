#!/usr/bin/env python3
import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    edges = data[2:]
    # reach[i][j] = True if i ->* j (i is stronger than j by known info)
    reach = [[False]*N for _ in range(N)]
    # Read edges
    idx = 0
    for _ in range(M):
        a = int(edges[idx]) - 1; b = int(edges[idx+1]) - 1
        idx += 2
        reach[a][b] = True
    # Floyd-Warshall for transitive closure
    for k in range(N):
        row_k = reach[k]
        for i in range(N):
            if reach[i][k]:
                # if i->k then for every j that k->j, set i->j
                for j in range(N):
                    if row_k[j]:
                        reach[i][j] = True
    # Count how many nodes j have no i != j with reach[i][j] == True
    candidates = []
    for j in range(N):
        ok = True
        for i in range(N):
            if i != j and reach[i][j]:
                ok = False
                break
        if ok:
            candidates.append(j+1)  # store 1-based
    # If exactly one candidate, print it; else -1
    if len(candidates) == 1:
        print(candidates[0])
    else:
        print(-1)

if __name__ == "__main__":
    main()