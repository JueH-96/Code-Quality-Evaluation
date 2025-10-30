import sys
import threading

def main():
    import sys

    data = sys.stdin
    n = int(data.readline().strip())
    # dp0 = max tastiness ending with healthy stomach
    # dp1 = max tastiness ending with upset stomach
    dp0 = 0
    dp1 = -10**40  # effectively negative infinity

    for _ in range(n):
        line = data.readline().split()
        x = int(line[0])
        y = int(line[1])
        # skipping keeps the same dp0, dp1
        nd0 = dp0
        nd1 = dp1
        if x == 0:
            # antidotal: from healthy or upset to healthy
            # eating gives dp0 + y or dp1 + y
            v0 = dp0 + y
            v1 = dp1 + y
            if v0 > nd0:
                nd0 = v0
            if v1 > nd0:
                nd0 = v1
        else:
            # poisonous: only from healthy -> upset
            v = dp0 + y
            if v > nd1:
                nd1 = v

        dp0, dp1 = nd0, nd1

    # He must survive (either healthy or upset). Also can choose to eat nothing -> 0.
    ans = dp0 if dp0 > dp1 else dp1
    if ans < 0:
        ans = 0
    print(ans)

if __name__ == "__main__":
    main()