MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    
    conditions = []
    for _ in range(M):
        L = int(input[ptr])
        ptr += 1
        R = int(input[ptr])
        ptr += 1
        X = int(input[ptr])
        ptr += 1
        if L <= X <= R:
            conditions.append((L, R, X))
    
    from collections import defaultdict

    min_R = [float('inf')] * (N + 2)
    S = defaultdict(list)
    for L, R, X in conditions:
        S[X].append((L, R))
    
    for x in S:
        min_r = min(r for l, r in S[x])
        min_R[x] = min_r
    min_R = [min_R[x] for x in range(N + 2)]
    
    valid = [False] * (N + 2)
    for x in range(1, N + 1):
        if not S[x]:
            valid[x] = True
            min_R[x] = N + 1
        else:
            upper = min(min_R[x], x + 1)
            if upper < 1:
                print(0)
                return
            valid[x] = upper >= 1
    
    max_n = N
    factorial = [1] * (max_n + 2)
    for i in range(1, max_n + 2):
        factorial[i] = factorial[i - 1] * i % MOD
    
    perm = [[0] * (max_n + 2) for _ in range(max_n + 2)]
    for n in range(max_n + 2):
        perm[n][0] = 1
        for k in range(1, n + 1):
            perm[n][k] = perm[n][k - 1] * (n - k + 1) % MOD
    
    dp_prev = defaultdict(int)
    dp_prev[N + 1] = 1
    
    for x in range(1, N + 1):
        dp_current = defaultdict(int)
        for k_prev in dp_prev:
            for k_x in range(1, min(min_R[x], x + 1) + 1):
                if k_x > k_prev:
                    continue
                if not valid[x]:
                    if k_x > N + 1:
                        continue
                    ways = 1
                else:
                    n = N - k_x + 1
                    k = N - x
                    if k < 0 or n < k:
                        ways = 0
                    else:
                        ways = perm[n][k]
                dp_current[k_x] = (dp_current[k_x] + dp_prev[k_prev] * ways) % MOD
        if not dp_current:
            print(0)
            return
        dp_prev = dp_current
    
    print(sum(dp_prev.values()) % MOD)

if __name__ == "__main__":
    main()