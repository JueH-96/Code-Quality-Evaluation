# YOUR CODE HERE
import sys

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    N,K = map(int, sys.stdin.readline().split())
    S = sys.stdin.read().strip()
    MOD = 998244353

    # Precompute palindrome masks
    pal_set = set()
    for mask in range(1<<K):
        is_palin = True
        for i in range(K//2):
            if ((mask >>i)&1) != ((mask >>(K-1-i))&1):
                is_palin = False
                break
        if is_palin:
            pal_set.add(mask)

    # Initialize DP
    dp_size = 1 << (K-1)
    current_dp = [0]*dp_size
    current_dp[0] =1

    for pos in range(N):
        next_dp = [0]*dp_size
        c_char = S[pos]
        if c_char == '?':
            options = [0,1]
        else:
            options = [0] if c_char == 'A' else [1]
        for s in range(dp_size):
            cnt = current_dp[s]
            if cnt ==0:
                continue
            for c in options:
                new_s = ((s <<1) |c) & ((1 <<(K-1)) -1)
                if pos >= K-1:
                    k_bits = (s <<1) |c
                    if k_bits not in pal_set:
                        next_dp[new_s] = (next_dp[new_s] + cnt) % MOD
                else:
                    next_dp[new_s] = (next_dp[new_s] + cnt) % MOD
        current_dp = next_dp

    result = sum(current_dp) % MOD
    print(result)

if __name__ == "__main__":
    main()