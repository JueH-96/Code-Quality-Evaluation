def main():
    import sys
    N_S_M_L = sys.stdin.read().split()
    N = int(N_S_M_L[0])
    S = int(N_S_M_L[1])
    M = int(N_S_M_L[2])
    L = int(N_S_M_L[3])
    
    packs = [(6, S), (8, M), (12, L)]
    max_pack_size = 12
    max_eggs = N + max_pack_size
    INF = float('inf')
    dp = [INF] * (max_eggs + 1)
    dp[0] = 0
    
    for size, cost in packs:
        for i in range(size, max_eggs + 1):
            dp[i] = min(dp[i], dp[i - size] + cost)
    
    min_cost = min(dp[N:max_eggs + 1])
    print(min_cost)

if __name__ == '__main__':
    main()