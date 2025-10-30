import sys
import threading

def main():
    import sys
    input = sys.stdin.readline
    N, X, Y = map(int, input().split())
    items = [tuple(map(int, input().split())) for _ in range(N)]
    INF = Y + 1  # any salt sum > Y we treat as INF
    
    # dp[k][s] = minimal total saltiness to achieve total sweetness s using exactly k dishes
    # we only care about dp[k][s] <= Y
    dp = [ [INF] * (X+1) for _ in range(N+1) ]
    dp[0][0] = 0
    
    # track maximum k for which dp[k] has any valid entry so far
    maxk = 0
    
    for A, B in items:
        # we will update dp[k+1] from dp[k], for k from maxk down to 0
        # after this item, maxk increases by at most 1
        lim = min(maxk, N-1)
        for k in range(lim, -1, -1):
            dp_k = dp[k]
            dp_k1 = dp[k+1]
            # only sweetness sums up to X-A can transition
            up = X - A
            # loop over possible sweetness sums
            # if dp_k[s] <= Y, we can take this dish
            for s in range(up+1):
                sb = dp_k[s]
                if sb <= Y:
                    nsb = sb + B
                    if nsb < dp_k1[s + A]:
                        # if new salt sum exceeds Y, mark INF
                        dp_k1[s + A] = nsb if nsb <= Y else INF
        # increase maxk by 1 (at most N)
        if maxk < N:
            maxk += 1

    # find largest k such that there exists a sweetness s with dp[k][s] <= Y
    ans = 0
    for k in range(N, -1, -1):
        row = dp[k]
        # scan row for any valid salt sum
        ok = False
        for s in range(X+1):
            if row[s] <= Y:
                ok = True
                break
        if ok:
            ans = k
            break

    print(ans)

if __name__ == "__main__":
    main()