import sys
import threading
def main():
    import sys
    data = sys.stdin.readline().split()
    if not data:
        return
    N = int(data[0])
    S = sys.stdin.readline().strip()
    C_list = list(map(int, sys.stdin.readline().split()))
    # Build bit array for S and cost array C indexed from 1
    sbit = [0] * (N + 2)
    for i, ch in enumerate(S, start=1):
        sbit[i] = 1 if ch == '1' else 0
    C = [0] + C_list
    # Prefix costs for the two global alternating patterns
    pre0 = [0] * (N + 2)
    pre1 = [0] * (N + 2)
    for j in range(1, N + 1):
        pj = j & 1
        exp0 = pj ^ 1  # expected bit if start=0
        exp1 = pj      # expected bit if start=1
        bit = sbit[j]
        cost = C[j]
        pre0[j] = pre0[j-1] + (cost if bit != exp0 else 0)
        pre1[j] = pre1[j-1] + (cost if bit != exp1 else 0)
    # Suffix costs for the two global patterns
    suf0 = [0] * (N + 3)
    suf1 = [0] * (N + 3)
    for j in range(N, 0, -1):
        pj = j & 1
        exp0 = pj ^ 1
        exp1 = pj
        bit = sbit[j]
        cost = C[j]
        suf0[j] = suf0[j+1] + (cost if bit != exp0 else 0)
        suf1[j] = suf1[j+1] + (cost if bit != exp1 else 0)
    # Try all split positions i = 1..N-1 and both starting bits s=0,1
    INF = 10**30
    ans = INF
    for i in range(1, N):
        pj_i = i & 1
        exp0_i = pj_i ^ 1
        exp1_i = pj_i
        pj_next = (i + 1) & 1
        # case s = 0
        cost_pre = pre0[i]
        s2 = exp0_i
        flag = s2 ^ pj_next
        cost_suf = suf0[i+1] if flag else suf1[i+1]
        tot = cost_pre + cost_suf
        if tot < ans:
            ans = tot
        # case s = 1
        cost_pre = pre1[i]
        s2 = exp1_i
        flag = s2 ^ pj_next
        cost_suf = suf0[i+1] if flag else suf1[i+1]
        tot = cost_pre + cost_suf
        if tot < ans:
            ans = tot
    print(ans)

if __name__ == "__main__":
    main()