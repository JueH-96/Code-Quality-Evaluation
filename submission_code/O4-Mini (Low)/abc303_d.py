import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    X = int(data[0])
    Y = int(data[1])
    Z = int(data[2])
    S = data[3].strip()
    INF = 10**30
    # dp0 = min cost to have produced up to i with caps-lock OFF
    # dp1 = same with caps-lock ON
    dp0 = 0
    dp1 = INF
    for ch in S:
        ndp0 = INF
        ndp1 = INF
        # produce next char = ch
        # consider previous state p=0(off) -> new state s
        # s = 0: k = 0 toggles
        #   costToggle = 0
        #   now in s=0
        #   pressing 'a' gives 'a' at cost X
        #   pressing Shift+'a' gives 'A' at cost Y
        if dp0 < INF:
            # p=0 -> s=0, k=0
            if ch == 'a':
                cost = dp0 + X
                if cost < ndp0: ndp0 = cost
            if ch == 'A':
                cost = dp0 + Y
                if cost < ndp0: ndp0 = cost
            # p=0 -> s=1, k=1 toggle
            # costToggle = Z
            base1 = dp0 + Z
            # now s=1: pressing 'a' gives 'A', pressing Shift+'a' gives 'a'
            if ch == 'A':
                cost = base1 + X
                if cost < ndp1: ndp1 = cost
            if ch == 'a':
                cost = base1 + Y
                if cost < ndp1: ndp1 = cost
        # consider previous state p=1(on)
        if dp1 < INF:
            # p=1 -> s=1, k=0
            if ch == 'A':
                cost = dp1 + X
                if cost < ndp1: ndp1 = cost
            if ch == 'a':
                cost = dp1 + Y
                if cost < ndp1: ndp1 = cost
            # p=1 -> s=0, k=1 toggle
            base0 = dp1 + Z
            if ch == 'a':
                cost = base0 + X
                if cost < ndp0: ndp0 = cost
            if ch == 'A':
                cost = base0 + Y
                if cost < ndp0: ndp0 = cost
        dp0, dp1 = ndp0, ndp1
    # At end, no need to reset caps; take min of both
    ans = dp0 if dp0 < dp1 else dp1
    print(ans)

if __name__ == "__main__":
    main()