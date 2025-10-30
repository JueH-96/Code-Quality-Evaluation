def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    MOD = 998244353
    
    N = int(data[0])
    matches = []
    index = 1
    for _ in range(N-1):
        p = int(data[index])
        q = int(data[index+1])
        matches.append((p, q))
        index += 2
    
    parent = [i for i in range(N+1)]
    size = [1] * (N+1)
    addition = [0] * (N+1)
    rank = [0] * (N+1)
    
    def find(x):
        if parent[x] != x:
            orig_parent = parent[x]
            parent[x] = find(parent[x])
            addition[x] += addition[orig_parent]
        return parent[x]
    
    for p, q in matches:
        root_p = find(p)
        root_q = find(q)
        a = size[root_p]
        b = size[root_q]
        
        # Compute a/(a+b) and b/(a+b) modulo MOD
        inv = pow(a + b, MOD-2, MOD)
        prob_p = a * inv % MOD
        prob_q = b * inv % MOD
        
        addition[root_p] = (addition[root_p] + prob_p) % MOD
        addition[root_q] = (addition[root_q] + prob_q) % MOD
        
        if rank[root_p] < rank[root_q]:
            parent[root_p] = root_q
            size[root_q] += size[root_p]
        else:
            parent[root_q] = root_p
            size[root_p] += size[root_q]
            if rank[root_p] == rank[root_q]:
                rank[root_p] += 1
    
    E = [0] * (N+1)
    for i in range(1, N+1):
        root = find(i)
        E[i] = addition[i] % MOD
    
    print(' '.join(str(E[i]) for i in range(1, N+1)))

if __name__ == "__main__":
    main()