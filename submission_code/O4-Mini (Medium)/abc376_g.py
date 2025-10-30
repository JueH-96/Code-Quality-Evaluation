import sys
import threading

def main():
    import sys
    data = sys.stdin
    MOD = 998244353
    inv_cache = {}
    def modinv(x):
        # Fermat's little theorem since MOD prime
        return pow(x, MOD-2, MOD)
    T = int(data.readline())
    out = []
    # Process each test case
    for _ in range(T):
        line = data.readline().strip()
        while line == "":
            line = data.readline().strip()
        N = int(line)
        # read parents p[1..N]
        p = list(map(int, data.readline().split()))
        # read a[1..N]
        a_list = list(map(int, data.readline().split()))
        # Build children lists for nodes 0..N
        children = [[] for __ in range(N+1)]
        for i in range(1, N+1):
            pi = p[i-1]
            children[pi].append(i)
        # Compute subtree weights w[i] = sum of a[j] in subtree of i
        # Use the fact p_i < i to do a reverse accumulation
        w = [0] * (N+1)
        # assign leaves
        for i in range(1, N+1):
            w[i] = a_list[i-1]
        # accumulate from N down to 1
        for i in range(N, 0, -1):
            pi = p[i-1]
            w[pi] += w[i]
        # total weight W = sum of a_i
        W = w[0]
        # Priority queue (max-heap via negative keys)
        import heapq
        pq = []
        # initially push children of root (node 0)
        for c in children[0]:
            # push pair (-w[c], c)
            heapq.heappush(pq, (-w[c], c))
        # simulate search order
        t = 1
        sum_mod = 0
        # for each node popped, accumulate a[node] * t
        while pq:
            negw, u = heapq.heappop(pq)
            # accumulate
            sum_mod = (sum_mod + a_list[u-1] * t) % MOD
            t += 1
            # push its children
            for v in children[u]:
                heapq.heappush(pq, (-w[v], v))
        # compute answer = sum_mod * inv(W) mod
        # cache inverses if repeated W
        if W not in inv_cache:
            inv_cache[W] = modinv(W)
        ans = sum_mod * inv_cache[W] % MOD
        out.append(str(ans))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()