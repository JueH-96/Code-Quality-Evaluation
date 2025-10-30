import sys
import threading

def main():
    import sys
    data = sys.stdin
    line = data.readline().strip()
    if not line:
        return
    n = int(line)
    # dp0: max tastiness ending with healthy stomach
    # dp1: max tastiness ending with upset stomach
    # Initialize dp0 = 0 (eat nothing), dp1 = -INF (impossible)
    INF = 10**30
    dp0 = 0
    dp1 = -INF

    for _ in range(n):
        xi, yi = data.readline().split()
        x = int(xi)
        y = int(yi)
        prev0, prev1 = dp0, dp1
        if x == 0:
            # antidote: eating from either state -> healthy
            # skip keeps states
            eat0_from_healthy = prev0 + y
            eat0_from_upset = prev1 + y
            dp0 = max(prev0, eat0_from_healthy, eat0_from_upset)
            # dp1 stays by skipping
            dp1 = prev1
        else:
            # poison: can only eat when healthy -> upset
            eat1 = prev0 + y
            # skip keeps upset state or choose to eat
            dp1 = max(prev1, eat1)
            # healthy state only by skipping
            dp0 = prev0

    # He must survive, so he can't be dead; dp0>=0 always, dp1 may be -INF or negative
    # If eating nothing is best, dp0 == 0; answer is max(dp0, dp1, 0)
    ans = dp0 if dp0 >= dp1 else dp1
    if ans < 0:
        ans = 0
    print(ans)

if __name__ == "__main__":
    main()