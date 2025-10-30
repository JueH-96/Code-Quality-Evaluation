def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Faster output
    import sys
    write = sys.stdout.write
    
    # Parse inputs
    # Format:
    #   N T
    #   S_1
    #   S_2
    #   ...
    #   S_N
    #
    # We want to count the number of pairs (i, j) so that T is a subsequence of S_i + S_j.
    
    # Strategy:
    # 1) Let T have length L. Precompute nextpos_forward for T:
    #    nextpos_forward[p][c] = the smallest index >= p in T where T[index] == c,
    #    or -1 if no such index.  We treat p as an index in [0..L], so if p == L,
    #    we say there's no place to match anything (all are -1).
    #
    # 2) For each string S_i, if we read T from 0 to end trying to match chars of S_i,
    #    we end up at some position endpos_i in [0..L].  (endpos_i == L means S_i alone
    #    already fully matches T.)
    #    Count how many S_i yield each possible endpos_i (call this freq[endpos_i]).
    #
    # 3) We also want to know for each string S_j whether, if we start trying to match
    #    T from some p in [0..L], we can reach L (i.e., fully match the remainder).
    #    Instead of building a big table, note that "S_j can complete T from p" is
    #    equivalent to "the reverse of S_j can match the reverse of T from 0..(L-p)".
    #
    # 4) So define T_rev = reverse(T).  Build nextpos_backward for T_rev similarly.
    #    Then for each string S_j (read in reverse order of chars), find how many chars
    #    of T_rev it can match from the start.  Call that endpos_rev_j.  We then know
    #    that if endpos_rev_j >= L - p, S_j can complete T from p.  We collect a frequency
    #    array freq_rev[k] = number of strings S_j whose reversed subsequence match length
    #    is exactly k.  Then the number of S_j that can complete from p is
    #    sum_{k >= L-p} freq_rev[k].
    #
    # 5) Prepare a suffix-sum G(k) = sum_{u >= k} freq_rev[u].  Then canComplete(p) = G(L-p).
    #
    # 6) Finally, the answer is:
    #         sum_{p=0}^{L-1} [ freq[p] * canComplete(p) ] + freq[L] * N
    #    Because if endpos_i = L, T is already matched by S_i, so any j works (N choices).
    
    # Be mindful of performance and memory.  We'll implement this carefully.
    sys.setrecursionlimit(10**7)
    
    # Read N and T
    # input_data = everything split
    idx = 0
    N = int(input_data[idx]); idx+=1
    T = input_data[idx]; idx+=1
    S_list = input_data[idx: idx+N]
    idx += N
    
    L = len(T)
    
    # Edge case: if T is empty (though not stated in constraints, just in case),
    # then T is always a subsequence.  The number of pairs is N^2.
    # But from constraints, T has length >= 1, so we skip that edge.
    
    # Build nextpos for T in forward direction
    # nextpos_forward[p][c] = next index >= p where T[next] == c, or -1
    # We'll store it in a 2D list of length L+1, each sublist has 26 ints.
    # Then we iterate from the end of T forward.
    
    # If L is large (up to 500k), we must be mindful of memory/time.
    # Implementation detail: we will build it by scanning from the end,
    # reusing arrays as we go.
    
    # Initialize with -1
    # We'll convert characters to integers: 'a' -> 0, ..., 'z' -> 25
    ALPH = 26
    ord_a = ord('a')
    
    nextpos_forward = [[-1]*ALPH for _ in range(L+1)]
    # For p = L, everything is -1
    
    # Fill from the back
    if L > 0:
        # Copy for p = L-1 from the row p+1, then fix the char T[L-1].
        # We'll do it in a loop from L-1 down to 0.
        last = [-1]*ALPH
        for c in range(ALPH):
            nextpos_forward[L][c] = -1
        for p in range(L-1, -1, -1):
            # copy from last
            for c in range(ALPH):
                nextpos_forward[p][c] = nextpos_forward[p+1][c]
            # fix the char T[p]
            c_idx = ord(T[p]) - ord_a
            nextpos_forward[p][c_idx] = p
    
    # We'll also build nextpos_backward for T_rev
    # T_rev = T[::-1], length L
    # nextpos_backward[p][c] = next index >= p in T_rev where T_rev[next] == c, or -1
    # Then for a string's reversed version, we can see how many chars of T_rev are matched.
    # Implementation is the same, just we pass T_rev to the same logic.
    T_rev = T[::-1]
    
    nextpos_backward = [[-1]*ALPH for _ in range(L+1)]
    
    if L > 0:
        for c in range(ALPH):
            nextpos_backward[L][c] = -1
        for p in range(L-1, -1, -1):
            for c in range(ALPH):
                nextpos_backward[p][c] = nextpos_backward[p+1][c]
            c_idx = ord(T_rev[p]) - ord_a
            nextpos_backward[p][c_idx] = p
    
    # We'll keep freq array for how many S_i endpos == p
    freq = [0]*(L+1)
    # We'll keep freq_rev array for how many S_i (in reverse) can match p chars of T_rev
    freq_rev = [0]*(L+1)
    
    # Function to find endpos in T if we start from 0
    def subsequence_endpos_forward(s):
        pos = 0  # index in T we've matched so far
        for ch in s:
            c_idx = ord(ch) - ord_a
            # nextpos_forward[pos][c_idx] is the first match of ch at or after pos
            nxt = nextpos_forward[pos][c_idx]
            if nxt == -1:
                # can't match any further
                return pos
            # otherwise we matched T[nxt], so move to nxt+1
            pos = nxt+1
            if pos > L:
                break
        return pos
    
    # Function to find how many chars of T_rev we match from the start
    def subsequence_endpos_backward(s):
        # We'll match T_rev from p=0 with the reversed string s
        # but we don't want to physically reverse s each time; we can just read s backwards
        pos = 0
        # s in normal order is S_i, we want to match T_rev with reversed(S_i).
        # So let's iterate s from end to start
        for i in range(len(s)-1, -1, -1):
            ch = s[i]
            c_idx = ord(ch) - ord_a
            nxt = nextpos_backward[pos][c_idx]
            if nxt == -1:
                return pos
            pos = nxt + 1
            if pos > L:
                break
        return pos
    
    # Read each S_i, compute forward endpos and backward endpos
    # Summation of lengths of S_i <= 5e5, so total time complexity is still manageable.
    for s in S_list:
        p_fwd = subsequence_endpos_forward(s)
        freq[p_fwd] += 1
        p_bwd = subsequence_endpos_backward(s)
        freq_rev[p_bwd] += 1
    
    # Now build the suffix sum G(k) = sum_{u >= k} freq_rev[u]
    # We'll store it in G array of length L+1, G[L] = freq_rev[L], then in decreasing order
    G = [0]*(L+1)
    running = 0
    for k in range(L, -1, -1):
        if k == L:
            running = freq_rev[L]
            G[L] = running
        elif k < L:
            running += freq_rev[k]
            G[k] = running
    
    # We have freq[0..L], G[0..L], G(k) = sum_{u >= k} freq_rev[u]
    # canComplete(p) = G(L-p), for p in [0..L].
    # The formula for total pairs:
    #   sum_{p=0}^{L-1} freq[p] * G(L-p) + freq[L]*N
    # We'll use a 64-bit integer (Python's int is unbounded).
    
    ans = 0
    # sum for p in [0..L-1]
    for p in range(L):
        if freq[p] > 0:
            ans += freq[p] * G[L-p]
    # plus freq[L] * N
    ans += freq[L] * N
    
    write(str(ans) + "
")

# Don't forget to call main!
if __name__ == "__main__":
    main()