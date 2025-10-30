import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:] ))
    MOD = 998244353
    MAX_SUM = 10
    MASKS = 1 << (MAX_SUM + 1)  # bits 0..10
    
    # Precompute transitions: for each mask and v=1..10, 
    # what new mask = mask | (mask<<v) (capped to bits 0..10).
    trans = [[0] * (MAX_SUM + 1) for _ in range(MASKS)]
    mask_limit = MASKS - 1
    for mask in range(MASKS):
        for v in range(1, MAX_SUM + 1):
            shifted = (mask << v) & mask_limit
            trans[mask][v] = mask | shifted
    
    # dp[mask] = number of ways (outcomes) so that the reachable-subset-sums
    # mask = mask of possible sums in [0..10].
    dp = [0] * MASKS
    dp[1] = 1  # initially only sum 0 is reachable
    
    for Ai in A:
        B = Ai if Ai <= MAX_SUM else MAX_SUM
        C = Ai - B  # number of face values > MAX_SUM
        nxt = [0] * MASKS
        if C > 0:
            # contributions where the large face value doesn't change reachable sums
            for mask in range(MASKS):
                v = dp[mask]
                if v:
                    nxt[mask] = (nxt[mask] + v * C) % MOD
        else:
            # C == 0, skip that part
            pass
        
        # contributions from face values 1..B
        for mask in range(MASKS):
            vdp = dp[mask]
            if not vdp:
                continue
            # for each small face outcome
            for v in range(1, B + 1):
                nm = trans[mask][v]
                nxt[nm] += vdp
                if nxt[nm] >= MOD:
                    nxt[nm] -= MOD
        dp = nxt
    
    # sum up all dp[mask] where sum=10 is reachable i.e. bit 10 is set
    target_bit = 1 << MAX_SUM
    favorable = 0
    for mask in range(MASKS):
        if mask & target_bit:
            favorable += dp[mask]
            if favorable >= MOD:
                favorable -= MOD
    
    # total number of outcomes
    total = 1
    for Ai in A:
        total = (total * Ai) % MOD
    
    # result = favorable * inv(total) mod MOD
    inv_total = pow(total, MOD-2, MOD)
    ans = favorable * inv_total % MOD
    print(ans)

if __name__ == "__main__":
    main()