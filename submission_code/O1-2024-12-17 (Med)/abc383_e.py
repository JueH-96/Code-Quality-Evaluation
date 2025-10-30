def main():
    import sys
    sys.setrecursionlimit(10**7)

    input_data = sys.stdin.read().strip().split()
    N, M, K = map(int, input_data[:3])
    ptr = 3
    edges = []
    for _ in range(M):
        u = int(input_data[ptr]); v = int(input_data[ptr+1]); w = int(input_data[ptr+2])
        ptr += 3
        edges.append((w,u,v))
    A = list(map(int, input_data[ptr:ptr+K]))
    ptr += K
    B = list(map(int, input_data[ptr:ptr+K]))
    ptr += K

    # Count how many times each vertex appears in A or B.
    # It's guaranteed by the problem that no vertex is both in A and in B.
    a_count = [0]*(N+1)
    b_count = [0]*(N+1)
    for x in A:
        a_count[x] += 1
    for x in B:
        b_count[x] += 1

    # Sort edges by ascending weight
    edges.sort(key=lambda x:x[0])

    # DSU (Disjoint Set Union) with extra data: how many unmatched A's and B's in each component
    parent = list(range(N+1))
    rank = [0]*(N+1)
    unmatchedA = a_count[:]  # how many A's are currently unmatched in the component
    unmatchedB = b_count[:]  # how many B's are currently unmatched in the component

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(r1, r2, w):
        # r1, r2 are different roots, merge them
        nonlocal total_matches, answer
        a1 = unmatchedA[r1]
        b1 = unmatchedB[r1]
        a2 = unmatchedA[r2]
        b2 = unmatchedB[r2]

        # Match cross pairs that become connected at weight w
        # cross 1: A in r1 with B in r2
        new_matches1 = min(a1, b2)
        a1 -= new_matches1
        b2 -= new_matches1

        # cross 2: A in r2 with B in r1
        new_matches2 = min(a2, b1)
        a2 -= new_matches2
        b1 -= new_matches2

        new_matches = new_matches1 + new_matches2
        if new_matches > 0:
            answer += new_matches * w
            total_matches += new_matches

        # DSU merge
        # attach smaller rank root to larger rank root
        if rank[r1] < rank[r2]:
            parent[r1] = r2
            unmatchedA[r2] = a1 + a2
            unmatchedB[r2] = b1 + b2
            return r2
        else:
            parent[r2] = r1
            unmatchedA[r1] = a1 + a2
            unmatchedB[r1] = b1 + b2
            if rank[r1] == rank[r2]:
                rank[r1] += 1
            return r1

    total_matches = 0
    answer = 0

    # Kruskal-like process in ascending order of edge weights
    for w, u, v in edges:
        ru = find(u)
        rv = find(v)
        if ru != rv:
            union(ru, rv, w)
            if total_matches == K:
                break

    print(answer)

# Don't forget to call main!
if __name__ == "__main__":
    main()