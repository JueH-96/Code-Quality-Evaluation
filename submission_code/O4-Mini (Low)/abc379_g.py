import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10000)
    M = 998244353
    input = sys.stdin.readline

    H, W = map(int, input().split())
    S = [list(input().strip()) for _ in range(H)]
    # Ensure W is the smaller dimension for DP width
    if H < W:
        # transpose S
        T = [['']*H for _ in range(W)]
        for i in range(H):
            for j in range(W):
                T[j][i] = S[i][j]
        S = T
        H, W = W, H

    # Now H >= W, we'll do DP over H rows of width W.
    # Pre-generate all "proper" row colorings of length W (no horizontal equal)
    # Colors: 0,1,2 for '1','2','3'
    all_states = []
    state_arr = []

    def dfs(pos, prev_color, cur_arr):
        if pos == W:
            # encode as integer in base 3
            code = 0
            for c in cur_arr:
                code = code * 3 + c
            all_states.append(code)
            state_arr.append(cur_arr[:])
            return
        for c in (0,1,2):
            if c != prev_color:
                cur_arr.append(c)
                dfs(pos+1, c, cur_arr)
                cur_arr.pop()

    dfs(0, -1, [])

    m = len(all_states)
    # Precompute adjacency: for each state index i, list of j such that
    # state_arr[i][k] != state_arr[j][k] for all k
    adj = [[] for _ in range(m)]
    for i in range(m):
        ai = state_arr[i]
        for j in range(m):
            aj = state_arr[j]
            ok = True
            for k in range(W):
                if ai[k] == aj[k]:
                    ok = False
                    break
            if ok:
                adj[i].append(j)

    # For each row, build list of valid state indices that match fixed chars
    row_valid = []
    for i in range(H):
        fixed = S[i]
        valid = []
        for idx in range(m):
            arr = state_arr[idx]
            ok = True
            for k in range(W):
                ch = fixed[k]
                if ch != '?' and arr[k] != (ord(ch)-ord('1')):
                    ok = False
                    break
            if ok:
                valid.append(idx)
        row_valid.append(valid)

    # DP arrays
    dp = [0]*m
    # Initialize with first row
    for idx in row_valid[0]:
        dp[idx] = 1

    # For each next row
    for i in range(1, H):
        valid = row_valid[i]
        ok_mask = [0]*m
        for idx in valid:
            ok_mask[idx] = 1
        dp2 = [0]*m
        # for each prev state j with dp[j]>0
        for j in range(m):
            v = dp[j]
            if v:
                for k in adj[j]:
                    if ok_mask[k]:
                        dp2[k] = (dp2[k] + v) % M
        dp = dp2

    # sum over last row valid states
    ans = 0
    for idx in row_valid[-1]:
        ans = (ans + dp[idx]) % M
    print(ans)

if __name__ == "__main__":
    main()