import sys
import threading

def main():
    import sys
    data = sys.stdin
    line = data.readline().split()
    if not line:
        print(-1)
        return
    N = int(line[0]); M = int(line[1])
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u,v,z = data.readline().split()
        u = int(u)-1; v = int(v)-1; z = int(z)
        adj[u].append((v, z))
        adj[v].append((u, z))
    from collections import deque
    visited = [False]*N
    val = [0]*N   # val[i] = XOR from component-root to i
    A = [0]*N     # final answer
    # we'll use up to 31 bits (0..30)
    BITS = 31

    for i in range(N):
        if not visited[i]:
            # BFS this component
            q = deque([i])
            visited[i] = True
            val[i] = 0
            comp_nodes = [i]
            ok = True
            while q:
                u = q.popleft()
                for v, z in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        val[v] = val[u] ^ z
                        q.append(v)
                        comp_nodes.append(v)
                    else:
                        # check consistency
                        if val[v] != (val[u] ^ z):
                            print(-1)
                            return
            # compute bit counts in this component
            K = len(comp_nodes)
            counts1 = [0]*BITS
            for u in comp_nodes:
                x = val[u]
                # count bits
                for b in range(BITS):
                    if (x >> b) & 1:
                        counts1[b] += 1
            # decide R_comp bits
            R = 0
            for b in range(BITS):
                c1 = counts1[b]
                c0 = K - c1
                # if inverting yields fewer 1s, invert => set R bit = 1
                if c1 > c0:
                    R |= (1 << b)
            # compute final A for nodes
            for u in comp_nodes:
                A[u] = val[u] ^ R

    # output
    out = sys.stdout
    out.write(" ".join(str(x) for x in A))

if __name__ == "__main__":
    main()