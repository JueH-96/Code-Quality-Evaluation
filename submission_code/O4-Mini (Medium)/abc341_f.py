import sys
import threading
def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it))
    g = [[] for _ in range(n)]
    for _ in range(m):
        u = int(next(it)) - 1; v = int(next(it)) - 1
        g[u].append(v)
        g[v].append(u)
    W = [0]*n
    for i in range(n):
        W[i] = int(next(it))
    A = [0]*n
    for i in range(n):
        A[i] = int(next(it))
    # Prepare vertices sorted by weight
    order = list(range(n))
    order.sort(key=lambda x: W[x])
    # f[i] = max number of operations from a single piece at i
    f = [0]*n
    # For each vertex in increasing weight, do knapsack on neighbors with smaller weight
    for x in order:
        cap = W[x] - 1
        # collect items: neighbors y with W[y] < W[x]
        items = []
        for y in g[x]:
            wy = W[y]
            if wy < W[x]:
                items.append((wy, f[y]))
        if cap <= 0 or not items:
            # can't put any neighbor, only the removal operation
            f[x] = 1
            continue
        # knapsack: capacity = cap, items with (weight, value)
        dp = [0] * (cap + 1)
        best = 0
        # for speed, bind locally
        dp_loc = dp
        for w_i, v_i in items:
            # if w_i > cap, skip
            if w_i > cap:
                continue
            # iterate backwards
            # localize for speed
            wi = w_i; vi = v_i
            for c in range(cap, wi-1, -1):
                nv = dp_loc[c-wi] + vi
                if nv > dp_loc[c]:
                    dp_loc[c] = nv
                    if nv > best:
                        best = nv
        # best is the max sum of f[y] we can get
        f[x] = best + 1
    # answer = sum A[i] * f[i]
    ans = 0
    # use 64-bit
    for i in range(n):
        if A[i]:
            ans += A[i] * f[i]
    print(ans)

if __name__ == "__main__":
    main()