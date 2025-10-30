# YOUR CODE HERE
import sys

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    NODES = N
    strong = [[False]*NODES for _ in range(NODES)]  # strong[i][j]=True if i is stronger than j
    for _ in range(M):
        Ai, Bi = map(int, sys.stdin.readline().split())
        Ai -= 1  # 0-based index
        Bi -= 1
        strong[Ai][Bi] = True

    # Floyd-Warshall algorithm for transitive closure
    for k in range(NODES):
        for i in range(NODES):
            if strong[i][k]:
                for j in range(NODES):
                    if strong[k][j]:
                        if not strong[i][j]:
                            strong[i][j] = True

    strongest_candidates = []
    for i in range(NODES):
        is_stronger_than_all = True
        for j in range(NODES):
            if i != j:
                if not strong[i][j]:
                    is_stronger_than_all = False
                    break
        if is_stronger_than_all:
            strongest_candidates.append(i+1)  # Convert back to 1-based index

    if len(strongest_candidates) == 1:
        print(strongest_candidates[0])
    else:
        print(-1)

if __name__ == "__main__":
    main()