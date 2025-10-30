import sys
import threading
def main():
    import sys
    input = sys.stdin.readline
    MOD = 998244353
    
    N, M, K = map(int, input().split())
    X = [0]*M
    Y = [0]*M
    for i in range(M):
        x,y = map(int, input().split())
        # zero‐based indices for convenience
        X[i] = x-1
        Y[i] = y-1

    # arr represents the dp vector: dp[v] = arr[(v + offset) % N]
    arr = [0]*N
    arr[0] = 1  # start at vertex 1 => index 0
    offset = 0

    for _ in range(K):
        # gather the contributions from "extra" edges before the cycle shift
        vals = [0]*M
        base = offset
        for i in range(M):
            # old dp at X[i]
            vals[i] = arr[(X[i] + base) % N]
        # apply the cycle edge shift: dp_next[v] = dp[v-1]
        offset = (offset + 1) % N
        # now add the extra‐edge contributions
        base = offset
        for i in range(M):
            arr[(Y[i] + base) % N] = (arr[(Y[i] + base) % N] + vals[i]) % MOD

    # sum of dp after K steps is sum of arr
    ans = sum(arr) % MOD
    print(ans)

if __name__ == "__main__":
    main()