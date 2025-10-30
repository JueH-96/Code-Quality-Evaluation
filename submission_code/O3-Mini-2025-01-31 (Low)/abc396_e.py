def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    
    # Build an undirected graph where each edge (u,v) is labeled with z meaning:
    # A[u] xor A[v] = z.
    graph = [[] for _ in range(N+1)]
    for _ in range(M):
        u = int(next(it))
        v = int(next(it))
        z = int(next(it))
        graph[u].append((v, z))
        graph[v].append((u, z))
    
    # f[i] will be our temporary value for vertex i, computed relative to one free parameter.
    # Later, for each connected component we can adjust all f[] by an XOR with some integer b.
    f = [None] * (N+1)
    # A[i] will hold the final answer for vertex i.
    A = [None] * (N+1)
    # visited array for DFS.
    visited = [False] * (N+1)
    
    # For each connected component, we will assign f using DFS.
    # Notice that if a vertex is isolated (i.e. not in any constraint),
    # then by default we can set its f[...] = 0.
    for i in range(1, N+1):
        if not visited[i]:
            component = []  # List of vertices in this component.
            stack = [i]
            f[i] = 0
            visited[i] = True
            while stack:
                u = stack.pop()
                component.append(u)
                for v, z in graph[u]:
                    # Propagate the relation: if A[u] xor A[v] = z then, once
                    # we have chosen f[u] (a partial assignment), we assign f[v] = f[u] xor z.
                    if not visited[v]:
                        visited[v] = True
                        f[v] = f[u] ^ z
                        stack.append(v)
                    else:
                        # If v has already been assigned, check the consistency:
                        # It must be that f[v] == f[u] xor z.
                        if f[v] != (f[u] ^ z):
                            sys.stdout.write("-1")
                            return
            
            # Now, for the connected component, we have a candidate assignment f[],
            # but we are allowed to XOR all values in this component by an arbitrary constant b.
            # Our final values will be: A[u] = f[u] xor b.
            # Because: (f[u] xor b) xor (f[v] xor b) = f[u] xor f[v] and the equation holds.
            # To minimize the sum, we want each A[u] to be as small as possible.
            # We can choose b (bitwise) independently for each bit.
            #
            # Consider one bit (2^bit) and let mask = 1 << bit.
            # For vertices in the component, count how many f[u] have that bit set and not set:
            #   - If we choose b's bit = 0, then for each vertex with that bit set, the corresponding bit in A[u]
            #     becomes 1; the cost for this bit would be count1 * (1 << bit).
            #   - If we choose b's bit = 1, then for each vertex with that bit not set, the corresponding bit in A[u]
            #     becomes 1; the cost would be count0 * (1 << bit).
            #
            # We choose b's bit = 0 when count1 <= count0, otherwise choose it as 1.
            b = 0
            # Consider up to 32 bits (this covers all values up to at least 1e9).
            for bit in range(32):
                mask = 1 << bit
                count0 = 0
                count1 = 0
                for u in component:
                    if f[u] & mask:
                        count1 += 1
                    else:
                        count0 += 1
                # We want to minimize the total sum contribution for this bit.
                # If count1 is larger than count0, then toggling this bit (setting b's bit=1) will yield
                # a cost of count0 * mask (which is less than count1 * mask).
                if count1 > count0:
                    b |= mask
            
            # With b chosen optimally for the component, assign final values.
            for u in component:
                A[u] = f[u] ^ b
                
    # Output the resulting sequence A[1..N] as space separated integers.
    sys.stdout.write(" ".join(str(A[i]) for i in range(1, N+1)))
    
if __name__ == '__main__':
    main()