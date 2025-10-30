MOD = 998244353

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    a_list = list(map(int, data[1:1+n]))
    
    fact = [1] * (n + 1)
    inv_fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i-1] * i % MOD
        
    inv_fact[n] = pow(fact[n], MOD - 2, MOD)
    for i in range(n, 0, -1):
        inv_fact[i-1] = inv_fact[i] * i % MOD
        
    children = [[] for _ in range(n)]
    for i in range(1, n):
        parent = a_list[i]
        children[parent].append(i)
        
    size = [0] * n
    dp = [0] * n
    
    def dfs(u):
        size[u] = 1
        total_ways = 1
        child_count = len(children[u])
        for v in children[u]:
            dfs(v)
            size[u] += size[v]
            total_ways = total_ways * dp[v] % MOD
            total_ways = total_ways * inv_fact[size[v] - 1] % MOD
            
        total_ways = total_ways * fact[size[u] - 1] % MOD
        total_ways = total_ways * inv_fact[child_count] % MOD
        dp[u] = total_ways
        
    dfs(0)
    print(dp[0])

if __name__ == '__main__':
    main()