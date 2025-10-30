MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    matches = []
    for _ in range(N-1):
        p = int(data[idx])
        q = int(data[idx+1])
        matches.append((p, q))
        idx += 2
    
    parent = [i for i in range(N+1)]
    size = [1] * (N+1)
    sum_e = [0] * (N+1)
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    
    for p, q in matches:
        root_p = find(p)
        root_q = find(q)
        if root_p == root_q:
            continue
        s1 = size[root_p]
        s2 = size[root_q]
        sum_s = s1 + s2
        inv_sum_s = pow(sum_s, MOD-2, MOD)
        a = (s1 * inv_sum_s) % MOD
        b = (s2 * inv_sum_s) % MOD
        
        sum_e[root_p] = (sum_e[root_p] + a * s1) % MOD
        sum_e[root_q] = (sum_e[root_q] + b * s2) % MOD
        
        if size[root_p] < size[root_q]:
            parent[root_p] = root_q
            size[root_q] += size[root_p]
            sum_e[root_q] = (sum_e[root_q] + sum_e[root_p]) % MOD
        else:
            parent[root_q] = root_p
            size[root_p] += size[root_q]
            sum_e[root_p] = (sum_e[root_p] + sum_e[root_q]) % MOD
    
    root = find(1)
    for i in range(1, N+1):
        current_root = find(i)
        e_i = sum_e[current_root] * pow(size[current_root], MOD-2, MOD) % MOD
        print(e_i, end=' ')
    print()

if __name__ == '__main__':
    main()