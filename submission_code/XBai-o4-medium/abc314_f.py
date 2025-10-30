import sys
MOD = 998244353

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    
    parent = list(range(N+1))  # 0..N
    size = [1] * (N+1)
    elements = [[] for _ in range(N+1)]
    for i in range(1, N+1):
        elements[i].append(i)
    
    ans = [0] * (N+1)
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    
    for _ in range(N-1):
        p = int(input[ptr])
        q = int(input[ptr+1])
        ptr += 2
        
        root_p = find(p)
        root_q = find(q)
        
        a = size[root_p]
        b = size[root_q]
        denominator = a + b
        inv_denominator = pow(denominator, MOD-2, MOD)
        probA = a * inv_denominator % MOD
        probB = b * inv_denominator % MOD
        
        # Add probA to all in elements[root_p]
        for x in elements[root_p]:
            ans[x] = (ans[x] + probA) % MOD
        
        # Add probB to all in elements[root_q]
        for x in elements[root_q]:
            ans[x] = (ans[x] + probB) % MOD
        
        # Merge the two sets
        if size[root_p] > size[root_q]:
            big_root = root_p
            small_root = root_q
        else:
            big_root = root_q
            small_root = root_p
        
        # Append small to big
        elements[big_root].extend(elements[small_root])
        parent[small_root] = big_root
        size[big_root] += size[small_root]
    
    print(' '.join(map(str, ans[1:N+1])))

if __name__ == "__main__":
    main()