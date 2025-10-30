import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    t = int(data[0])
    ans = []
    idx = 1
    for _ in range(t):
        n = int(data[idx]); idx += 1
        A = list(map(int, data[idx:idx+n]))
        idx += n
        # compress into runs B
        B = []
        prev = None
        for x in A:
            if x != prev:
                B.append(x)
                prev = x
        L = len(B)
        if L < 3:
            ans.append(str(L))
            continue
        # dp[i] = max savings in first i runs
        # we only need last 3 values to save memory
        dp0 = 0  # dp[i-3]
        dp1 = 0  # dp[i-2]
        dp2 = 0  # dp[i-1]
        # we will compute dp3 for i=3..L
        for i in range(3, L+1):
            # check B[i-3] == B[i-1]
            if B[i-3] == B[i-1]:
                dp3 = dp0 + 1
                if dp2 > dp3:
                    dp3 = dp2
            else:
                dp3 = dp2
            # shift
            dp0, dp1, dp2 = dp1, dp2, dp3
        saves = dp2  # this is dp[L]
        res = L - saves
        ans.append(str(res))
    sys.stdout.write("
".join(ans))

if __name__ == "__main__":
    main()