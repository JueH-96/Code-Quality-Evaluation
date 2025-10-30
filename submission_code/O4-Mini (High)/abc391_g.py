def main():
    import sys
    from collections import deque
    input = sys.stdin.readline
    mod = 998244353

    # Read input
    N, M = map(int, input().split())
    S = input().strip()

    # BFS over the DP-states of the LCS-array f[1..N].
    # A state is a tuple f1..fN where f[i] = LCS length of S[:i] vs current T-prefix.
    state_to_id = {}
    states = []          # list of tuples, each length N
    transitions = []     # transitions[s][c] = next state id on letter c

    # initial state: f[i]=0 for all i
    init = tuple([0]*N)
    state_to_id[init] = 0
    states.append(init)
    transitions.append(None)

    q = deque([0])
    # Pre‐make the alphabet to avoid repeated chr() calls
    alphabet = [chr(ord('a')+i) for i in range(26)]

    while q:
        s1 = q.popleft()
        f1 = states[s1]  # tuple of length N
        # build f_prev = [0] + list(f1)
        f_prev = [0] * (N+1)
        for i in range(N):
            f_prev[i+1] = f1[i]

        # compute transitions for each letter
        trans = [0]*26
        for c_idx, c in enumerate(alphabet):
            # compute next DP-array f2
            f2 = [0]*(N+1)
            # f2[0] = 0
            for i in range(1, N+1):
                if S[i-1] == c:
                    # match case
                    val = f_prev[i-1] + 1
                else:
                    # no-match: take max of left or up
                    if f_prev[i] >= f2[i-1]:
                        val = f_prev[i]
                    else:
                        val = f2[i-1]
                f2[i] = val
            t2 = tuple(f2[1:])  # drop the zero‐th
            if t2 in state_to_id:
                s2 = state_to_id[t2]
            else:
                s2 = len(states)
                state_to_id[t2] = s2
                states.append(t2)
                transitions.append(None)
                q.append(s2)
            trans[c_idx] = s2

        transitions[s1] = trans

    S_cnt = len(states)

    # Build for each state a list of (next_state, count_of_letters_that_go_there)
    s2_counts = [None] * S_cnt
    for s1 in range(S_cnt):
        cnts = {}
        row = transitions[s1]
        for s2 in row:
            cnts[s2] = cnts.get(s2, 0) + 1
        # store as list of (state, count)
        s2_counts[s1] = list(cnts.items())

    # DP over length M of T: dp_cur[s] = number of T-prefixes landing in state s
    dp_cur = [0]*S_cnt
    dp_cur[0] = 1

    for _ in range(M):
        dp_next = [0]*S_cnt
        for s1 in range(S_cnt):
            v = dp_cur[s1]
            if not v:
                continue
            for s2, cnum in s2_counts[s1]:
                dp_next[s2] = (dp_next[s2] + v * cnum) % mod
        dp_cur = dp_next

    # Collect answers: final LCS length = f[N], which is states[s][N-1]
    ans = [0]*(N+1)
    for s in range(S_cnt):
        f_state = states[s]
        k = f_state[N-1]
        ans[k] = (ans[k] + dp_cur[s]) % mod

    # Output
    print(" ".join(str(x) for x in ans))


if __name__ == "__main__":
    main()