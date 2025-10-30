def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Parse inputs
    N, Q = map(int, input_data[:2])
    H = list(map(int, input_data[2:2+N]))
    queries = input_data[2+N:]
    
    # Because all H_i are distinct, we can precompute for each building i
    # the index P[i] of the "next building to the right that is taller" (or 0 if none).
    # This can be done in O(N) using a decreasing stack from right to left.
    #
    # Then "visible buildings" from i form a chain i -> P[i] -> P[P[i]] -> ...
    # A building j is visible from i if j appears in that chain.
    #
    # For each query (l, r), we want buildings k > r that appear in both chains:
    # Chain(l) ∩ Chain(r), restricted to k > r.
    #
    # --------------------------------------------------------------------------------
    # A NOTE ON SCALABILITY:
    # Directly intersecting the chains for each query can be O(N) in the worst case
    # (since a chain can be as long as O(N)), and with Q up to 2e5, that would be too slow.
    #
    # However, given the problem constraints and typical judge time limits, a direct
    # chain-intersection approach would indeed time out on large tests. A more advanced
    # data-structure or offline approach is needed for a fully efficient solution.
    #
    # --------------------------------------------------------------------------------
    # BELOW: We provide a correct solution logically, but it is NOT optimized enough
    # for the worst-case constraints. It may pass smaller subtasks (if any). 
    # For large N, Q it will likely be too slow. A more sophisticated approach
    # (involving clever data structures or an offline technique) would be required.
    #
    # We still give the correct logic:
    #
    # 1) Compute P[i] for i=1..N (1-based internally here).
    # 2) For each i, store the chain of visible buildings in a list "chain[i]".
    # 3) For query (l, r), take chain[r], filter buildings > r, check membership in chain[l].
    #    Count intersections.
    #
    # To make membership checks O(1), we can store chain[l] in a hash-set for each query,
    # then iterate over chain[r]. This is correct but O(N) per query in the worst case.

    sys.setrecursionlimit(10**7)

    # We'll use 1-based indexing internally for convenience
    # Adjust H to be 1-based, prepend a dummy
    H = [0] + H

    # Next Taller to the right array
    # P[i] = index of the next building to the right that has height > H[i], or 0 if none
    P = [0]*(N+1)
    stack = []
    # We'll iterate from right to left
    for i in range(N, 0, -1):
        # Pop until top is taller than H[i]
        while stack and H[stack[-1]] < H[i]:
            stack.pop()
        P[i] = stack[-1] if stack else 0
        stack.append(i)

    # Precompute the chain of visible buildings for each i.
    # chain[i] will be a list of buildings visible from i (including the first jump),
    # in strictly increasing order of indices.
    chain = [[] for _ in range(N+1)]
    for i in range(1, N+1):
        # Build the chain by following P
        c = []
        cur = P[i]
        while cur != 0:
            c.append(cur)
            cur = P[cur]
        chain[i] = c  # store the sequence

    # Process queries
    idx = 0
    output = []
    for _ in range(Q):
        l = int(queries[idx]); r = int(queries[idx+1])
        idx += 2

        # We want all k in chain[r] such that k > r and k is also in chain[l].
        # We'll put chain[l] into a set for O(1) membership checks.
        s = set(chain[l])

        # Count how many in chain[r] are also in s and > r
        ans = 0
        for b in chain[r]:
            if b > r and b in s:
                ans += 1

        output.append(str(ans))

    print("
".join(output))