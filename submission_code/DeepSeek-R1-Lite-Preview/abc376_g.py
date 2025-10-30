MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    T = int(data[idx])
    idx += 1
    for _ in range(T):
        N = int(data[idx])
        idx += 1
        p = list(map(int, data[idx:idx+N]))
        idx += N
        a = list(map(int, data[idx:idx+N]))
        idx += N
        children = [[] for _ in range(N+1)]
        for i in range(1, N+1):
            children[p[i-1]].append(i)
        S = sum(a)
        S_v = [0] * (N+1)
        def dfs_S(v):
            if v >= 1:
                S_v[v] = a[v-1]
            else:
                S_v[v] = 0
            for u in children[v]:
                dfs_S(u)
                S_v[v] += S_v[u]
        dfs_S(0)
        E = [0] * (N+1)
        def dfs_E(v):
            if not children[v]:
                if v >= 1:
                    E[v] = a[v-1] * pow(S, MOD-2, MOD) % MOD
                else:
                    E[v] = 0
                return
            sorted_children = sorted(children[v], key=lambda u: -S_v[u])
            sum_temp = 0
            j = 1
            for u in sorted_children:
                dfs_E(u)
                sum_temp = (sum_temp + S_v[u] * (j + E[u])) % MOD
                j += 1
            E[v] = sum_temp * pow(S_v[v], MOD-2, MOD) % MOD
        dfs_E(0)
        E0 = E[0]
        print(E0 % MOD)

if __name__ == '__main__':
    main()