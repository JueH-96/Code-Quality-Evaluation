import sys
import threading
def main():
    import sys
    from collections import deque
    sys.setrecursionlimit(1000000)
    mod = 998244353

    data = sys.stdin.read().split()
    N = int(data[0]); M = int(data[1]); S = data[2]

    # Build all reachable LCS-DP rows via BFS
    # State f is tuple of length N+1, f[0]=0
    start = tuple([0] * (N+1))
    state_index = {start: 0}
    states = [start]
    q = deque([start])

    while q:
        f = q.popleft()
        idx = state_index[f]
        # try each possible next letter c
        for ci in range(26):
            # compute next DP row f'
            # f'[0]=0
            fp = [0] * (N+1)
            # fill for j=1..N
            # use local variables for speed
            prev = f  # tuple
            # we need prev[j], prev[j-1]
            ch = chr(ord('a') + ci)
            for j in range(1, N+1):
                v = prev[j]
                if S[j-1] == ch:
                    # can extend from prev[j-1]
                    v2 = prev[j-1] + 1
                    if v2 > v:
                        v = v2
                fp[j] = v
            fp_t = tuple(fp)
            if fp_t not in state_index:
                state_index[fp_t] = len(states)
                states.append(fp_t)
                q.append(fp_t)

    num_states = len(states)
    # Precompute transitions counts: from each state s, a dict mapping s2 -> count of letters
    next_counts = [dict() for _ in range(num_states)]
    for s_idx, f in enumerate(states):
        cnt = {}
        # for each letter
        for ci in range(26):
            ch = chr(ord('a') + ci)
            fp = [0] * (N+1)
            prev = f
            for j in range(1, N+1):
                v = prev[j]
                if S[j-1] == ch:
                    v2 = prev[j-1] + 1
                    if v2 > v:
                        v = v2
                fp[j] = v
            fp_t = tuple(fp)
            s2 = state_index[fp_t]
            cnt[s2] = cnt.get(s2, 0) + 1
        next_counts[s_idx] = cnt

    # DP over length M
    dp = [0] * num_states
    dp[0] = 1  # start at state 0
    for _ in range(M):
        ndp = [0] * num_states
        for s in range(num_states):
            v = dp[s]
            if v:
                for s2, c in next_counts[s].items():
                    ndp[s2] = (ndp[s2] + v * c) % mod
        dp = ndp

    # Collect answers by final f[N]
    ans = [0] * (N+1)
    for s_idx, f in enumerate(states):
        k = f[N]
        ans[k] = (ans[k] + dp[s_idx]) % mod

    print(" ".join(str(ans[k]) for k in range(N+1)))

if __name__ == "__main__":
    main()