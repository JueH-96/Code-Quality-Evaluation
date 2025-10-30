import sys
from functools import lru_cache

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    P = int(input[1])
    
    k = N // 2
    
    max_n_comb = k * k
    
    # Precompute comb_mod
    max_comb = max_n_comb
    comb_mod = [[0]*(max_comb +1) for _ in range(max_comb +1)]
    for n in range(max_comb +1):
        comb_mod[n][0] = 1
        if n >= 1:
            comb_mod[n][n] = 1
        for m in range(1, n):
            comb_mod[n][m] = (comb_mod[n-1][m-1] + comb_mod[n-1][m]) % P
    
    @lru_cache(maxsize=None)
    def dp(a, b, m):
        if a == 1 and b == 1:
            if m == 1:
                return 1 % P
            else:
                return 0
        elif a == 1:
            if m < 1 or m > b:
                return 0
            return comb_mod[b][m]
        elif b == 1:
            if m < 1 or m > a:
                return 0
            return comb_mod[a][m]
        max_edges = a * b
        if m < 0 or m > max_edges:
            return 0
        total = comb_mod[max_edges][m]
        for a1 in range(1, a):
            for b1 in range(1, b):
                a_rest = a - a1
                b_rest = b - b1
                max_remaining_edges = a_rest * b_rest
                x_min = max(0, m - max_remaining_edges)
                x_max = min(a1 * b1, m)
                ways_choose_a = comb_mod[a][a1]
                ways_choose_b = comb_mod[b][b1]
                ways_choose = (ways_choose_a * ways_choose_b) % P
                for x in range(x_min, x_max + 1):
                    conn = dp(a1, b1, x)
                    if conn == 0:
                        continue
                    remaining_m = m - x
                    ways_remaining = comb_mod[max_remaining_edges][remaining_m]
                    term = (ways_choose * conn) % P
                    term = (term * ways_remaining) % P
                    total = (total - term) % P
        return total % P
    
    # Now compute answers for each M
    M_start = N - 1
    M_end = N * (N - 1) // 2
    res = []
    # Precompute comb_mod for the bipart count
    if N - 1 >= k - 1:
        bipart = comb_mod[N-1][k-1]
    else:
        bipart = 0
    for M in range(M_start, M_end + 1):
        if M > (k * k):
            res.append(0)
        else:
            cnt = dp(k, k, M)
            ans = (bipart * cnt) % P
            res.append(ans)
    print(' '.join(map(str, res)))

if __name__ == "__main__":
    main()