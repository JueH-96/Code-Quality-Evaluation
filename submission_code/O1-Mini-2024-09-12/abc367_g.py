# YOUR CODE HERE
import sys
import sys
import sys
from collections import defaultdict
import sys

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    MOD = 998244353
    N, M, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    
    # Group counts by value
    from collections import Counter
    cnt = Counter(A)
    unique_vals = list(cnt.keys())
    counts = list(cnt.values())
    H = len(unique_vals)
    
    # Initialize DP
    dp = [defaultdict(int) for _ in range(M)]
    dp[0][0] = 1
    for idx in range(H):
        v = unique_vals[idx]
        c = counts[idx]
        # Precompute ways to choose k elements from c
        # and their contribution to size modulo M and XOR
        # For each possible k, compute number of ways to choose k
        # and whether k is even or odd for XOR contribution
        comb = [1] * (c +1)
        for i in range(1, c+1):
            comb[i] = comb[i-1] * (c - i +1) * pow(i, MOD-2, MOD) % MOD
        # Calculate even and odd counts with size modulo M
        even = [0] * M
        odd = [0] * M
        for k in range(c+1):
            ways = comb[k] * pow(2, c -k, MOD) % MOD
            size_mod = k % M
            if k %2 ==0:
                even[size_mod] = (even[size_mod] + ways) % MOD
            else:
                odd[size_mod] = (odd[size_mod] + ways) % MOD
        # Update DP
        new_dp = [defaultdict(int) for _ in range(M)]
        for j in range(M):
            for y in dp[j]:
                cnt_val = dp[j][y]
                # Even selection, XOR unchanged
                for sm in range(M):
                    nj = (j + sm) % M
                    ny = y
                    new_dp[nj][ny] = (new_dp[nj][ny] + cnt_val * even[sm]) % MOD
                # Odd selection, XOR flipped with v
                for sm in range(M):
                    nj = (j + sm) % M
                    ny = y ^ v
                    new_dp[nj][ny] = (new_dp[nj][ny] + cnt_val * odd[sm]) % MOD
        dp = new_dp
    
    # Now, dp[0] contains C[y] for size divisible by M
    C = dp[0]
    
    # Precompute y^K for all possible y
    max_y = max(C.keys()) if C else 0
    y_pow_k = {}
    for y in C:
        y_pow_k[y] = pow(y, K, MOD)
    
    # Compute the final sum
    answer = 0
    for y in C:
        answer = (answer + C[y] * y_pow_k[y]) % MOD
    print(answer)

if __name__ == "__main__":
    main()