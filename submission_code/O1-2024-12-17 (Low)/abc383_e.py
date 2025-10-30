def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, M, K = map(int, input_data[0:3])
    edges = []
    idx = 3
    for _ in range(M):
        u, v, w = map(int, input_data[idx:idx+3])
        idx += 3
        edges.append((w,u,v))
    A = list(map(int, input_data[idx:idx+K]))
    idx += K
    B = list(map(int, input_data[idx:idx+K]))
    idx += K

    # Disjoint Set (Union-Find) structure
    parent = list(range(N+1))
    rank_ = [0]*(N+1)
    # Store, for each root, how many A-tokens and B-tokens it has, and how many pairs matched so far
    # (Actually we only need sumOfA, sumOfB, plus (optionally) matchedPairs if needed.)
    sumA = [0]*(N+1)
    sumB = [0]*(N+1)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b, w):
        # merge sets of a and b, return how many new matches
        rootA = find(a)
        rootB = find(b)
        if rootA == rootB:
            return 0  # no new matches

        # union by rank
        if rank_[rootA] < rank_[rootB]:
            rootA, rootB = rootB, rootA
        parent[rootB] = rootA
        if rank_[rootA] == rank_[rootB]:
            rank_[rootA] += 1

        # sum up tokens
        newA = sumA[rootA] + sumA[rootB]
        newB = sumB[rootA] + sumB[rootB]
        oldMatchedA = min(sumA[rootA], sumB[rootA])  # matched inside rootA alone
        oldMatchedB = min(sumA[rootB], sumB[rootB])  # matched inside rootB alone
        newMatched = min(newA, newB)
        newMatches = newMatched - (oldMatchedA + oldMatchedB)

        # store back
        sumA[rootA] = newA
        sumB[rootA] = newB
        # we can set the child root's sumA, sumB to 0 if we like
        sumA[rootB] = 0
        sumB[rootB] = 0

        return newMatches

    # Count how many A/B tokens belong to each node
    for a in A:
        sumA[a] += 1
    for b in B:
        sumB[b] += 1

    # Sort edges by weight
    edges.sort(key=lambda x:x[0])

    matched_count = 0
    ans = 0

    # Kruskal-like process
    for w,u,v in edges:
        new_matches = union(u, v, w)
        if new_matches > 0:
            ans += w * new_matches
            matched_count += new_matches
            if matched_count == K:
                break

    print(ans)

# Do not forget to call main()
if __name__ == "__main__":
    main()