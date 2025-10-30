import sys
import threading
def main():
    import sys
    input = sys.stdin.readline
    N, M, X1 = map(int, input().split())
    A = [0]*M
    B = [0]*M
    S = [0]*M
    T = [0]*M
    for i in range(M):
        a,b,s,t = map(int, input().split())
        A[i] = a-1
        B[i] = b-1
        S[i] = s
        T[i] = t
    # Build per‐city arrival lists: for each city c, list of (T_i, i) where train i arrives at c.
    arrivals = [[] for _ in range(N)]
    for i in range(M):
        arrivals[B[i]].append((T[i], i))
    # Sort each arrival list by T_i ascending
    for c in range(N):
        if arrivals[c]:
            arrivals[c].sort()
    # Build list of train indices sorted by departure time S_i ascending
    trains_by_S = list(range(M))
    trains_by_S.sort(key=lambda i: S[i])
    # dp[i] = minimal "dist" for train i, i.e. sum of w's from train1 to i
    INF = 10**30
    dp = [INF]*M
    dp[0] = 0   # train 1 has dist 0
    # For each city c we keep:
    #   idx[c] = how many arrivals we've ingested (prefix of arrivals[c] with T_i <= current S_j)
    #   best[c] = minimum over ingested arrivals of (dp[i] - T_i)
    idx = [0]*N
    best = [INF]*N
    # Process trains in order of increasing S_j
    for j in trains_by_S:
        c_dep = A[j]
        # ingest all arrivals i at city c_dep with T_i <= S_j
        arr_list = arrivals[c_dep]
        k = idx[c_dep]
        # advance pointer while arrival time <= this departure time
        while k < len(arr_list) and arr_list[k][0] <= S[j]:
            Ti, i = arr_list[k]
            di = dp[i]
            if di < INF:
                # candidate dp[i] - T_i
                val = di - Ti
                if val < best[c_dep]:
                    best[c_dep] = val
            k += 1
        idx[c_dep] = k
        # now we can relax dp[j]
        b = best[c_dep]
        if b < INF:
            cand = b + S[j]
            if cand < dp[j]:
                dp[j] = cand
    # Now compute X_j = max(0, X1 - dp[j])
    # Output X_2 ... X_M
    out = []
    for j in range(1, M):
        dj = dp[j]
        if dj >= X1 or dj == INF:
            out.append("0")
        else:
            out.append(str(X1 - dj))
    sys.stdout.write(" ".join(out))

if __name__ == "__main__":
    main()