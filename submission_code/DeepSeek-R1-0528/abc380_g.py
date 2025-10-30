mod = 998244353

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    k = int(data[1])
    P = list(map(int, data[2:2+n]))
    
    size = n + 10
    tree = [0] * (size)
    
    def update(i, delta):
        while i <= n:
            tree[i] += delta
            i += i & -i
            
    def query(i):
        s = 0
        while i:
            s += tree[i]
            i -= i & -i
        return s
        
    def query_range(l, r):
        if l > r:
            return 0
        return query(r) - query(l-1)
    
    I0 = 0
    for i in range(n):
        val = P[i]
        a = query_range(val+1, n)
        I0 = (I0 + a) % mod
        update(val, 1)
    
    tree = [0] * (size)
    total_within = 0
    current = 0
    
    for i in range(0, k):
        val = P[i]
        a = query_range(val+1, n)
        current = (current + a) % mod
        update(val, 1)
    total_within = (total_within + current) % mod
    
    for L in range(0, n - k):
        val_remove = P[L]
        a1 = query(val_remove - 1)
        update(val_remove, -1)
        current = (current - a1) % mod
        
        val_add = P[L+k]
        a2 = query_range(val_add+1, n)
        current = (current + a2) % mod
        update(val_add, 1)
        total_within = (total_within + current) % mod
    
    T = n - k + 1
    inv4 = pow(4, mod-2, mod)
    term1 = (k * (k - 1)) % mod * inv4 % mod
    invT = pow(T, mod-2, mod)
    term2 = total_within * invT % mod
    ans = (I0 + term1 - term2) % mod
    if ans < 0:
        ans += mod
    print(ans)

if __name__ == "__main__":
    main()