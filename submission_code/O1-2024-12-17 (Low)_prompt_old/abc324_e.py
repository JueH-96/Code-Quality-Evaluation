def solve():
    import sys
    import bisect

    input_data = sys.stdin.read().strip().split()
    # input_data layout:
    # [0] = N, [1] = T, then S_1, S_2, ..., S_N
    N = int(input_data[0])
    T = input_data[1]
    S_list = input_data[2:]
    # T can be large, total length of all S_i is up to 5e5, N up to 5e5.

    m = len(T)
    # Edge case: if m == 0, every pair trivially contains the empty T as subsequence,
    # but problem constraints say T length >= 1, so we skip that case.

    # Precompute "nextPos" for T (forward matching).
    # nextPos[i][c] = smallest index >= i in T where character c occurs; = m if none
    # We'll build this from right to left.
    # For convenience, convert T to numeric [0..25].
    T_num = [ord(c) - ord('a') for c in T]
    # Create nextPos array of dimension m+1 by 26
    # nextPos[m][c] = m for all c
    nextPos = [[m]*(26) for _ in range(m+1)]
    last = [m]*26
    for i in range(m-1, -1, -1):
        last[T_num[i]] = i
        for c in range(26):
            nextPos[i][c] = last[c]
    for c in range(26):
        nextPos[m][c] = m

    # Precompute "nextPosRev" for reversed T (for matching suffixes).
    # We'll define T_rev similarly.
    T_rev_num = T_num[::-1]
    nextPosRev = [[m]*(26) for _ in range(m+1)]
    last_rev = [m]*26
    for i in range(m-1, -1, -1):
        last_rev[T_rev_num[i]] = i
        for c in range(26):
            nextPosRev[i][c] = last_rev[c]
    for c in range(26):
        nextPosRev[m][c] = m

    # We define two arrays:
    # f[i] = how many chars of T can be matched (from the start) by S_i
    # g[i] = how many chars of T can be matched (from the end) by S_i
    f = [0]*N
    g = [0]*N

    # Function to get forward match length for one string S
    def get_forward_match_length(S):
        """Return how many characters from T can be matched as a subsequence in S (from the start of T)."""
        p = 0  # pointer in T
        for ch in S:
            if p == m:  # already matched all T
                break
            c = ord(ch) - ord('a')
            if nextPos[p][c] < m:
                p = nextPos[p][c] + 1
            else:
                # cannot match further
                break
        return p  # number of matched characters

    # Function to get reverse match length for one string S
    # i.e., how many chars from the END of T can be matched if we read S from right to left
    def get_reverse_match_length(S):
        p = 0  # pointer in reversed T
        # We'll traverse S from right to left
        # T_rev_num is the reversed T
        for i in range(len(S)-1, -1, -1):
            if p == m:
                break
            c = ord(S[i]) - ord('a')
            if nextPosRev[p][c] < m:
                p = nextPosRev[p][c] + 1
            else:
                break
        return p

    # Read each S_i, compute f[i], g[i].
    # Recall total length of all S_i is at most 5e5, so this is feasible.
    for i in range(N):
        S = S_list[i]
        f[i] = get_forward_match_length(S)
        g[i] = get_reverse_match_length(S)

    # We want to count number of pairs (i, j) such that f[i] + g[j] >= m.
    # Sort g. For each f[i], let need = m - f[i]. We want count of j with g[j] >= need.
    # That is total_j - bisect_left(g, need).
    g.sort()
    ans = 0
    from bisect import bisect_left

    for i in range(N):
        need = m - f[i]
        if need <= 0:
            # Then any j works because we already matched >= m from S_i alone
            ans += N
        else:
            # We find how many g[j] >= need
            idx = bisect_left(g, need)
            ans += (N - idx)

    print(ans)