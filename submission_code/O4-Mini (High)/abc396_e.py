import sys
from collections import deque

def main():
    input = sys.stdin.readline
    data = input().split()
    if not data:
        return
    N = int(data[0])
    M = int(data[1])
    # build adjacency list
    adj = [[] for _ in range(N)]
    for _ in range(M):
        x, y, z = map(int, input().split())
        u = x - 1
        v = y - 1
        adj[u].append((v, z))
        adj[v].append((u, z))

    visited = [False] * N
    D = [0] * N      # D[u] = xor-distance from the component's root
    A = [0] * N      # final answers

    BIT = 30  # enough to cover bits up to 2^29 (~5e8) and 10^9<2^30

    for i in range(N):
        if not visited[i]:
            # BFS/DFS to label this component and check consistency
            comp = []
            dq = deque([i])
            visited[i] = True
            D[i] = 0
            comp.append(i)
            while dq:
                u = dq.popleft()
                for v, z in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        D[v] = D[u] ^ z
                        dq.append(v)
                        comp.append(v)
                    else:
                        # check consistency
                        if D[v] != (D[u] ^ z):
                            print(-1)
                            return

            # Now we have all D[u] for u in this component.
            # Choose x = A[root] to minimize sum_u (D[u] ^ x).
            # Do it bit by bit.
            k = len(comp)
            count1 = [0] * BIT
            for u in comp:
                du = D[u]
                # count bits of du
                for b in range(BIT):
                    if (du >> b) & 1:
                        count1[b] += 1

            x = 0
            # for each bit, decide x_b = 0 or 1
            for b in range(BIT):
                ones = count1[b]
                zeros = k - ones
                # if setting x_b=1 flips all zeros->1 and ones->0
                # contribution is (zeros)*(1<<b)
                # if x_b=0, contribution is (ones)*(1<<b)
                # pick the smaller
                if zeros < ones:
                    x |= (1 << b)

            # assign A[u]
            for u in comp:
                A[u] = D[u] ^ x

    # If we reach here, we have a valid assignment
    print(" ".join(str(val) for val in A))

if __name__ == "__main__":
    main()