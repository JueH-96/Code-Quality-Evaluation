import sys
import threading

def main():
    import sys
    data = sys.stdin
    line = data.readline()
    if not line:
        return
    N = int(line)
    # dp0 = max tastiness ending with healthy stomach
    # dp1 = max tastiness ending with upset stomach
    dp0 = 0
    # sentinel for impossible/upset-before-eating anything
    dp1 = -10**30

    for _ in range(N):
        x_str, y_str = data.readline().split()
        x = int(x_str)
        y = int(y_str)
        if x == 0:
            # antidotal course
            # skip it -> dp0 stays dp0, dp1 stays dp1
            # eat it -> from healthy: dp0 + y stays healthy
            #         from upset: dp1 + y goes to healthy
            # so new dp0 = max(old dp0, dp0+y, dp1+y)
            # new dp1 = old dp1 (no way to go to upset by eating antidote)
            new_dp0 = dp0
            t = dp0 + y
            if t > new_dp0:
                new_dp0 = t
            t = dp1 + y
            if t > new_dp0:
                new_dp0 = t
            new_dp1 = dp1
        else:
            # poisonous course
            # skip it -> dp0 stays, dp1 stays
            # eat -> from healthy: dp0 + y goes to upset
            # from upset: death (invalid)
            new_dp0 = dp0
            new_dp1 = dp1
            t = dp0 + y
            if t > new_dp1:
                new_dp1 = t

        dp0, dp1 = new_dp0, new_dp1

    # best is max of ending healthy or upset (dp0 >= 0 always)
    ans = dp0 if dp0 > dp1 else dp1
    print(ans)

if __name__ == "__main__":
    main()