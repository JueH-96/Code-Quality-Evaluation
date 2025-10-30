def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline
    
    N = int(input())
    
    # Adjacency list for the tree
    adj = [[] for _ in range(N + 1)]
    
    for _ in range(N - 1):
        A, B = map(int, input().split())
        adj[A].append(B)
        adj[B].append(A)
    
    # Read the C array
    C = list(map(int, input().split()))
    
    # Arrays to store subtree sum of C and the f(v) values
    subtreeC = [0] * (N + 1)
    fVal = [0] * (N + 1)
    
    # The total sum of all C values
    totalC = sum(C)
    
    # We'll compute f(1) (the weighted distance sum from vertex 1)
    # and also compute subtreeC for each vertex
    totalDistFromRoot = 0
    
    def dfs1(u, parent, depth):
        nonlocal totalDistFromRoot
        # Add the current node's value to its subtree sum
        subtreeC[u] = C[u - 1]
        # Accumulate distance * C[u-1]
        totalDistFromRoot += depth * C[u - 1]
        # Recur for children
        for v in adj[u]:
            if v == parent:
                continue
            dfs1(v, u, depth + 1)
            subtreeC[u] += subtreeC[v]
    
    # Re-rooting DFS: compute f(v) for each v using the known formula
    def dfs2(u, parent):
        for v in adj[u]:
            if v == parent:
                continue
            # When moving the root from u to its child v, the f value changes by:
            #    fVal[v] = fVal[u] + (totalC - 2 * subtreeC[v])
            fVal[v] = fVal[u] + (totalC - 2 * subtreeC[v])
            dfs2(v, u)
    
    # Arbitrarily choose 1 as the root
    dfs1(1, -1, 0)
    # f(1) is totalDistFromRoot
    fVal[1] = totalDistFromRoot
    
    # Compute f(v) for all vertices by rerooting
    dfs2(1, -1)
    
    # The answer is the minimum f(v) across all vertices
    print(min(fVal[1:]))

# Call main() to run the solution
if __name__ == "__main__":
    main()