import threading
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1000000)
    data = sys.stdin
    MOD = 998244353
    T = int(data.readline())
    out = []
    for _ in range(T):
        line = data.readline().strip()
        while line == "":
            line = data.readline().strip()
        N = int(line)
        p = [0]*(N+1)
        parts = data.readline().split()
        for i in range(1, N+1):
            p[i] = int(parts[i-1])
        parts = data.readline().split()
        a = [0]*(N+1)
        for i in range(1, N+1):
            a[i] = int(parts[i-1])
        # 1) compute subtree sums S using the fact that p_i < i
        S = a[:]  # copy
        for u in range(N, 0, -1):
            S[p[u]] += S[u]
        # 2) build children lists
        children = [[] for _ in range(N+1)]
        for i in range(1, N+1):
            children[p[i]].append(i)
        # 3) sort children of each node by subtree sum descending
        for u in range(N+1):
            if len(children[u])>1:
                children[u].sort(key=lambda x: S[x], reverse=True)
        # 4) do iterative DFS to assign positions and accumulate weighted sum
        total = S[0]  # sum a[1..N]
        inv_total = pow(total, MOD-2, MOD)
        cur_pos = 0
        weighted_sum = 0
        # stack entries: (node, next child index)
        stack = [(0, 0)]
        while stack:
            u, idx = stack.pop()
            if idx == 0:
                # entering u
                if u != 0:
                    cur_pos += 1
                    # add a[u] * cur_pos
                    weighted_sum = (weighted_sum + a[u] * cur_pos) % MOD
            if idx < len(children[u]):
                # still have children to visit
                # push this node back with next idx
                stack.append((u, idx+1))
                # push the child
                v = children[u][idx]
                stack.append((v, 0))
            # else, all children done, continue
        # result = weighted_sum / total mod
        ans = weighted_sum * inv_total % MOD
        out.append(str(ans))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()