MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    K = int(data[idx])
    idx += 1
    
    next_add = [[] for _ in range(N + 1)]  # 1-based indexing
    
    for _ in range(M):
        X = int(data[idx])
        idx += 1
        Y = int(data[idx])
        idx += 1
        next_add[X].append(Y)
    
    dp = {1: 1}
    
    for _ in range(K):
        new_dp = {}
        for u in dp:
            c = dp[u]
            # Handle cycle edge: u -> u+1 (mod N)
            if u < N:
                v_cycle = u + 1
            else:
                v_cycle = 1
            new_dp[v_cycle] = (new_dp.get(v_cycle, 0) + c) % MOD
            # Handle additional edges
            for v in next_add[u]:
                new_dp[v] = (new_dp.get(v, 0) + c) % MOD
        dp = new_dp
    
    total = sum(dp.values()) % MOD
    print(total)

if __name__ == '__main__':
    main()