import sys
import itertools

def read_input():
    N = int(sys.stdin.readline().strip())
    D = []
    for i in range(N-1):
        row = list(map(int, sys.stdin.readline().strip().split()))
        D.append(row)
    return N, D

def solve(N, D):
    max_weight = 0
    for r in range(1, N//2 + 1):
        for edges in itertools.combinations(range(N*(N-1)//2), r):
            used_vertices = set()
            weight = 0
            valid = True
            for edge_idx in edges:
                i = edge_idx // (N-1)
                j = edge_idx % (N-1)
                if j < i:
                    j += 1
                if i in used_vertices or j in used_vertices:
                    valid = False
                    break
                used_vertices.add(i)
                used_vertices.add(j)
                weight += D[i][j-1]
            if valid:
                max_weight = max(max_weight, weight)
    return max_weight

def main():
    N, D = read_input()
    result = solve(N, D)
    print(result)

if __name__ == "__main__":
    main()