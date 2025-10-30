MOD = 998244353

def pow_mod(x, y, p):
    res = 1
    x = x % p
    while y > 0:
        if y & 1:
            res = (res * x) % p
        y = y >> 1
        x = (x * x) % p
    return res

def inv_mod(x, p):
    return pow_mod(x, p - 2, p)

T = int(input())
for _ in range(T):
    N = int(input())
    parents = list(map(int, input().split()))
    a = list(map(int, input().split()))
    
    total_a = sum(a)
    
    children = [[] for _ in range(N + 1)]
    for i, p in enumerate(parents):
        children[p].append(i + 1)
    
    expected_ops = 0
    for i in range(1, N + 1):
        prob = (a[i-1] * inv_mod(total_a, MOD)) % MOD
        
        ops = 0
        path = []
        curr = i
        while curr != 0:
            path.append(curr)
            curr = parents[curr - 1]
        ops = len(path)
        expected_ops = (expected_ops + ops * prob) % MOD

    print(expected_ops)