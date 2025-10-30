import sys

MOD = 998244353

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr += 1
    M = int(input[ptr]); ptr += 1
    B = list(map(int, input[ptr:ptr+N]))
    ptr += N

    # Precompute post_ways
    post_ways = [1] * N
    for i in range(N-2, -1, -1):
        next_pos = i + 1
        if B[next_pos] == -1:
            mult = M
        else:
            mult = 1
        post_ways[i] = post_ways[i+1] * mult % MOD

    # Precompute min_fixed and k_var
    INF = M + 1
    min_fixed = [INF] * N
    k_var = [0] * N
    for i in range(1, N):
        prev_val = B[i-1]
        if prev_val == -1:
            current_min = min_fixed[i-1]
        else:
            current_min = min(min_fixed[i-1], prev_val)
        min_fixed[i] = current_min
        k_var[i] = k_var[i-1] + (1 if (prev_val == -1) else 0)

    total = 0
    for i in range(N):
        if i == 0:
            # Contribution for the first element
            if B[i] == -1:
                own = M
            else:
                own = 1
            contrib = own * post_ways[i] % MOD
            total = (total + contrib) % MOD
        else:
            current_min = min_fixed[i]
            k = k_var[i]
            if B[i] != -1:
                x = B[i]
                if current_min != INF and x >= current_min:
                    # No contribution
                    continue
                else:
                    base = M - x
                    ways_prev = pow(base, k, MOD)
                    contrib = ways_prev * post_ways[i] % MOD
                    total = (total + contrib) % MOD
            else:
                # Variable case
                if current_min == INF:
                    upper = M
                else:
                    upper = current_min - 1
                if upper < 1:
                    continue
                sum_S = 0
                for x in range(1, upper + 1):
                    base = M - x
                    term = pow(base, k, MOD)
                    sum_S = (sum_S + term) % MOD
                contrib = sum_S * post_ways[i] % MOD
                total = (total + contrib) % MOD

    print(total % MOD)

if __name__ == "__main__":
    main()