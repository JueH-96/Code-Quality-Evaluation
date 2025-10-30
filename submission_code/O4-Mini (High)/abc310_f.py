import sys
import threading
def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    MOD = 998244353
    # We track reachable sums 0..10 via 11-bit masks (bit s means sum s reachable).
    # dp[mask] = probability (mod MOD) to have exactly that reachable-set mask.
    # Initial: only sum 0 reachable -> mask = 1<<0 = 1
    MAXSUM = 10
    BITS = MAXSUM + 1  # 11
    FULLMASK = (1 << BITS) - 1  # 2047
    # Precompute transitions: for r in 1..10, for each mask b, what is new mask b' = b | ((b<<r)&FULLMASK)
    # b_r[r][b] = new mask
    b_r = [[0] * (1 << BITS) for _ in range(BITS)]
    for r in range(1, BITS):
        shift = r
        # For each old mask b
        # new mask = b OR ((b << r) & FULLMASK)
        br = b_r[r]
        for b in range(1 << BITS):
            br[b] = b | ((b << shift) & FULLMASK)
    # dp arrays
    size = 1 << BITS  # 2048
    dp = [0] * size
    dp[1] = 1  # mask=1 (only sum=0)
    # Main DP over dice
    for Ai in A:
        invAi = pow(Ai, MOD-2, MOD)
        low = Ai if Ai <= MAXSUM else MAXSUM
        high = Ai - low
        # coeff for staying same mask (r>10)
        coeff_stay = high * invAi % MOD
        coeff_move = invAi  # for each r in 1..low
        dp_new = [0] * size
        # local refs
        dp_old = dp
        dpn = dp_new
        br = b_r
        cs = coeff_stay
        cm = coeff_move
        # iterate over all masks
        for mask in range(size):
            p = dp_old[mask]
            if p:
                # stay
                if cs:
                    dpn[mask] = (dpn[mask] + p * cs) % MOD
                # moves for r=1..low
                # we do small loop
                for r in range(1, low+1):
                    m2 = br[r][mask]
                    dpn[m2] = (dpn[m2] + p * cm) % MOD
        dp = dp_new
    # Sum probabilities of masks where sum 10 is reachable (bit 10 set)
    bit10 = 1 << MAXSUM
    ans = 0
    for mask in range(size):
        if mask & bit10:
            ans = (ans + dp[mask]) % MOD
    print(ans)

if __name__ == "__main__":
    main()