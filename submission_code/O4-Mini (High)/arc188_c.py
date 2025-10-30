def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.readline().split()
    if not data:
        return
    N, M = map(int, data)
    # We'll create 2N variables: 
    #  x[0..N-1]  for the honesty (1=honest, 0=liar)
    #  p[0..N-1]  where p[A] = x[A] XOR y[A]  (y[A] is confusion[A])
    # Each testimony A_i, B_i, C_i gives the constraint
    #   p[A_i] XOR x[B_i] = C_i
    # which is a parity constraint on two variables modulo 2.
    # We solve it by a DSU with parity.
    parent = list(range(2*N))
    diff = [0] * (2*N)     # diff[u] = val[u] XOR val[parent[u]]
    size = [1] * (2*N)
    
    def find(u):
        # path‐compress, updating diff[u] to be parity to the root
        if parent[u] != u:
            orig = parent[u]
            root = find(orig)
            diff[u] ^= diff[orig]
            parent[u] = root
        return parent[u]
    
    def union(u, v, w):
        # enforce val[u] XOR val[v] = w
        ru = find(u)
        rv = find(v)
        if ru == rv:
            # already in same component: check consistency
            return (diff[u] ^ diff[v]) == w
        # otherwise link smaller tree under larger
        c = w ^ diff[u] ^ diff[v]  # this will become diff of the child root
        if size[ru] < size[rv]:
            parent[ru] = rv
            diff[ru] = c
            size[rv] += size[ru]
        else:
            parent[rv] = ru
            diff[rv] = c
            size[ru] += size[rv]
        return True
    
    # Read and apply all testimony constraints
    for _ in range(M):
        a, b, c = map(int, sys.stdin.readline().split())
        a -= 1
        b -= 1
        u = N + a  # p[a]
        v = b      # x[b]
        if not union(u, v, c):
            print(-1)
            return
    
    # Force a finalize/find on all nodes so diff[u] is parity to its root
    for u in range(2*N):
        find(u)
    
    # Reconstruct confusion: y[A] = p[A] XOR x[A]
    res = ['0'] * N
    for i in range(N):
        yi = diff[N + i] ^ diff[i]
        res[i] = '1' if yi else '0'
    
    print(''.join(res))


if __name__ == "__main__":
    main()