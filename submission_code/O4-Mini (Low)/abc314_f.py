import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline
    mod = 998244353

    def modinv(x):
        return pow(x, mod - 2, mod)

    N = int(input().strip())
    p = [0] * (N)
    q = [0] * (N)
    for i in range(N - 1):
        pi, qi = map(int, input().split())
        p[i] = pi
        q[i] = qi

    # DSU to build merge tree
    parent = list(range(2 * N))    # 1..2N-1
    size = [0] * (2 * N)
    for i in range(1, N + 1):
        size[i] = 1
    # children adjacency: for each internal node id>N, store list of (child, weight)
    children = [[] for _ in range(2 * N)]

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    # Build the merge tree
    # Internal nodes indices: N+1 ... N+(N-1) = 2N-1
    for i in range(N - 1):
        u = p[i]
        v = q[i]
        ru = find(u)
        rv = find(v)
        su = size[ru]
        sv = size[rv]
        tot = su + sv
        inv_tot = modinv(tot)
        # weights
        wu = su * inv_tot % mod
        wv = sv * inv_tot % mod
        newid = N + 1 + i
        # attach children
        # first child is the component of p[i], with weight P(first wins)=su/(su+sv)
        # second child is comp of q[i], weight sv/(su+sv)
        children[newid].append((ru, wu))
        children[newid].append((rv, wv))
        # union in DSU under newid
        parent[ru] = newid
        parent[rv] = newid
        size[newid] = tot

    # The final root is node 2N-1
    root = 2 * N - 1

    # DFS to accumulate sums
    ans = [0] * (N + 1)
    stack = [(root, 0)]
    while stack:
        node, acc = stack.pop()
        for (ch, w) in children[node]:
            acc_ch = (acc + w) % mod
            if ch <= N:
                ans[ch] = acc_ch
            else:
                stack.append((ch, acc_ch))

    # output
    out = " ".join(str(ans[i]) for i in range(1, N + 1))
    print(out)

if __name__ == "__main__":
    main()