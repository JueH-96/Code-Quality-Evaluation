def solve():
    n = int(input())
    matches = []
    for _ in range(n - 1):
        matches.append(list(map(int, input().split())))
    
    parent = list(range(n + 1))
    size = [1] * (n + 1)
    expected_wins = [0] * n
    mod = 998244353
    
    def find_set(v):
        if v == parent[v]:
            return v
        parent[v] = find_set(parent[v])
        return parent[v]
        
    def unite_sets(a, b):
        a = find_set(a)
        b = find_set(b)
        if a != b:
            parent[b] = a
            size[a] += size[b]
            
    def power(a, b):
        res = 1
        a %= mod
        while b > 0:
            if b % 2 == 1:
                res = (res * a) % mod
            a = (a * a) % mod
            b //= 2
        return res
        
    def inv(n):
        return power(n, mod - 2)
        
    def divide(num, den):
        return (num * inv(den)) % mod
        
    for p_i, q_i in matches:
        rep_p = find_set(p_i)
        rep_q = find_set(q_i)
        size_p = size[rep_p]
        size_q = size[rep_q]
        prob_p_wins = divide(size_p, size_p + size_q)
        prob_q_wins = divide(size_q, size_p + size_q)
        
        for j in range(1, n + 1):
            rep_j = find_set(j)
            if rep_j == rep_p:
                expected_wins[j-1] = (expected_wins[j-1] + prob_p_wins) % mod
            elif rep_j == rep_q:
                expected_wins[j-1] = (expected_wins[j-1] + prob_q_wins) % mod
                
        unite_sets(p_i, q_i)
        
    print(*(expected_wins))

if __name__ == '__main__':
    solve()