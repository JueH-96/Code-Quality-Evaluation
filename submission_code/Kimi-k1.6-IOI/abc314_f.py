MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    idx += 1
    
    parent = list(range(N+1))
    potential = [0] * (N+1)
    size = [1] * (N+1)
    value = [0] * (N+1)
    
    def find(u):
        path = []
        while parent[u] != u:
            path.append(u)
            u = parent[u]
        root = u
        path.reverse()
        for v in path:
            original_parent = parent[v]
            parent[v] = root
            potential[v] = (potential[v] + potential[original_parent]) % MOD
        return root
    
    for _ in range(N-1):
        p = int(data[idx])
        q = int(data[idx+1])
        idx +=2
        
        x = find(p)
        y = find(q)
        
        a = size[x]
        b = size[y]
        
        denominator = (a + b) % MOD
        inv_denominator = pow(denominator, MOD-2, MOD)
        
        probX = (a * inv_denominator) % MOD
        probY = (b * inv_denominator) % MOD
        
        value[x] = (value[x] + probX) % MOD
        value[y] = (value[y] + probY) % MOD
        
        if size[x] < size[y]:
            x, y = y, x
        
        parent[y] = x
        size[x] += size[y]
        potential[y] = (value[y] - value[x]) % MOD
    
    result = []
    for i in range(1, N+1):
        find(i)
        res = (value[parent[i]] + potential[i]) % MOD
        result.append(res)
    
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()