def main():
    import sys, bisect
    data = sys.stdin.buffer.read().splitlines()
    if not data:
        return
    # parse the first line
    header = data[0].decode().split()
    n = int(header[0])
    T = header[1]
    m = len(T)
    # read S strings
    S_list = [data[i].decode().rstrip('
') for i in range(1, n+1)]
    
    # Precompute the "next" table for T.
    # For any state p (which means “we have matched the first p letters of T”),
    # nxt[p][c] gives the new state after reading letter c.
    # If no occurrence exists, we set it to m (a state that means "we are done", because m is the maximum).
    # We create an array nxt[0..m] where each entry is a list of 26 integers.
    nxt = [[m] * 26 for _ in range(m+1)]
    # For state m (i.e. when we have already matched all letters), it remains m.
    # Now fill nxt backwards; for p from m-1 downto 0:
    nxt[m] = [m] * 26
    # Make a copy of nxt[m] for initialization
    for p in range(m-1, -1, -1):
        # first copy the next state from p+1
        nxt[p] = nxt[p+1][:]
        # update the entry for the letter T[p]:
        # When at state p, if we read T[p] then the new state becomes p+1.
        c = ord(T[p]) - 97
        nxt[p][c] = p+1

    # Define a simulation function.
    # Given a starting state 'state' (i.e. we have already matched state letters of T)
    # and a string s, simulate reading s and return the final state.
    # (Since the transition function is defined using the nxt-table, this simulation runs in O(|s|) time.)
    def simulate(s, state):
        cur = state
        for ch in s:
            cur = nxt[cur][ord(ch)-97]
        return cur

    # For every string S in S_list we want two pieces of information:
    #  1. x = simulate(S, 0): the final progress when using S alone starting with state 0.
    #  2. need = the minimal starting state p (0 <= p <= m) for which simulate(S, p) equals m.
    #     (If S is “powerful” enough that with some starting progress you can complete T with S, note
    #      the minimal required starting progress; if even starting from 0 does not complete T, we set need = INF.)
    INF = m + 1
    xs = [0] * n          # List of results x for each S.
    needs = [INF] * n     # List of needed starting progress for each S to yield T.

    # For each S, compute simulation from state 0.
    for i, s in enumerate(S_list):
        final_state = simulate(s, 0)
        xs[i] = final_state
        # If even starting from 0 we cannot match all of T, then no matter what p we use the result can’t be m.
        # We denote need = INF in that case.
        if final_state < m:
            needs[i] = INF
        else:
            # Otherwise, because the function p -> simulate(s, p) is monotonic
            # (i.e. if p1 <= p2 then simulate(s, p1) <= simulate(s, p2)),
            # we can binary search for the minimal p in [0, m] with simulate(s, p) == m.
            lo, hi = 0, m
            while lo < hi:
                mid = (lo + hi) // 2
                if simulate(s, mid) == m:
                    hi = mid
                else:
                    lo = mid + 1
            needs[i] = lo

    # Now notice that for the overall concatenation S_i + S_j,
    # the process is equivalent to first reading S_i (which brings progress x = simulate(S_i, 0))
    # and then reading S_j starting from that progress.
    # Thus S_i + S_j will contain T as a subsequence exactly when:
    #    simulate(S_j, simulate(S_i, 0)) == m.
    # For a fixed S_j, if we precomputed need_j as the minimal p that is required so that simulate(S_j, p)== m,
    # then S_i will work provided simulate(S_i, 0) >= need_j.
    #
    # Our answer is therefore:
    #    sum_{j with need_j != INF} (number of S_i with xs[i] >= need_j).
    
    xs.sort()
    ans = 0
    for need in needs:
        if need == INF:
            continue
        pos = bisect.bisect_left(xs, need)
        ans += (n - pos)
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()