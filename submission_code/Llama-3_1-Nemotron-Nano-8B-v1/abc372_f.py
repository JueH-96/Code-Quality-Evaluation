import bisect

MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr +=1
    M = int(input[ptr]); ptr +=1
    K = int(input[ptr]); ptr +=1

    E = [[] for _ in range(N+1)]  # 1-based
    for _ in range(M):
        X = int(input[ptr]); ptr +=1
        Y = int(input[ptr]); ptr +=1
        E[X].append(Y)

    S = set()
    for x in range(1, N+1):
        if len(E[x]) > 0:
            S.add(x)
    S = sorted(S)
    len_S = len(S)

    next_S = [0]*(N+1)
    d = [0]*(N+1)

    # Preprocess next_S and d for all nodes
    for u in range(1, N+1):
        if u in S:
            idx = (bisect.bisect_left(S, u) + 1) % len_S
            next_S[u] = S[idx]
            d[u] = (next_S[u] - u) % N
        else:
            idx = bisect.bisect_left(S, u)
            if idx < len_S:
                next_S[u] = S[idx]
            else:
                next_S[u] = S[0]
            d[u] = (next_S[u] - u) % N

    # Preprocess transitions for nodes in S
    transitions = [[] for _ in range(N+1)]
    for u in S:
        # Cycle edge
        v = next_S[u]
        cost = d[u]
        transitions[u].append( (v, cost) )
        # Extra edges
        for y in E[u]:
            if y in S:
                transitions[u].append( (y, 1) )
            else:
                vy = next_S[y]
                vyd = d[y]
                transitions[u].append( (vy, 1 + vyd) )

    # Initialize DP
    dp = [{} for _ in range(K+1)]
    if 1 in S:
        dp[0][1] = 1
    else:
        if d[1] > K:
            print(0)
            return
        dp[d[1]][next_S[1]] = 1

    # Process each step
    for s in range(K+1):
        if not dp[s]:
            continue
        for u in list(dp[s].keys()):
            cnt = dp[s][u]
            for (v, cost) in transitions[u]:
                new_s = s + cost
                if new_s > K:
                    continue
                if v in dp[new_s]:
                    dp[new_s][v] = (dp[new_s][v] + cnt) % MOD
                else:
                    dp[new_s][v] = cnt % MOD

    # Sum all possibilities in dp[K]
    res = 0
    for v in dp[K]:
        res = (res + dp[K][v]) % MOD
    print(res)

if __name__ == "__main__":
    main()