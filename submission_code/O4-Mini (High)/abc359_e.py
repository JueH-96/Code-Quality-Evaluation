import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    H = [int(next(it)) for _ in range(n)]
    # dp[i] = time when A[i]>0 for first time, 0-based i -> corresponds to A_{i+1}
    dp = [0] * n
    # monotonic stack of indices with H in non‐increasing order
    st = []
    for i in range(n):
        # pop smaller heights, we want nearest left j with H[j] >= H[i]
        while st and H[st[-1]] < H[i]:
            st.pop()
        if not st:
            # no previous barrier >= H[i], treat as j=0 sentinel
            # then dp[i] = (i+1)*H[i] + 1
            dp[i] = H[i] * (i + 1) + 1
        else:
            j = st[-1]
            # j < i and H[j] >= H[i]
            # dp[i] = dp[j] + (i-j)*H[i]
            dp[i] = dp[j] + H[i] * (i - j)
        st.append(i)
    # output
    out = sys.stdout
    out.write(" ".join(str(x) for x in dp))

if __name__ == "__main__":
    main()