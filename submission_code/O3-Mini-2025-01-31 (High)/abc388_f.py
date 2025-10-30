#!/usr/bin/env python3
def main():
    import sys, math
    from collections import deque
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    A = int(next(it))
    B = int(next(it))
 
    # --- Special-case: forced jumps when A == B. ---
    if A == B:
        # Then the only possible landing positions are: 1, 1+A, 1+2A, …, and N must equal 1+k*A.
        if (N - 1) % A != 0:
            sys.stdout.write("No")
            return
        k = (N - 1) // A
        forb = []
        for _ in range(M):
            Lf = int(next(it))
            Rf = int(next(it))
            forb.append((Lf, Rf))
        possible = True
        for (L_f, R_f) in forb:
            # Check if there is some jump i (1 <= i <= k) with landing = 1 + i*A in [L_f, R_f].
            i_min = (L_f - 1 + A - 1) // A  # ceil((L_f-1)/A)
            i_max = (R_f - 1) // A
            if i_min <= i_max and i_min <= k and i_max >= 1:
                possible = False
                break
        sys.stdout.write("Yes" if possible else "No")
        return
 
    # --- Now the case A < B ---
    forb = []
    for _ in range(M):
        Lf = int(next(it))
        Rf = int(next(it))
        forb.append((Lf, Rf))
 
    # Compute safe intervals S (the complement of forbidden intervals in [1,N])
    safe = []
    if M == 0:
        safe.append((1, N))
    else:
        if forb[0][0] > 1:
            safe.append((1, forb[0][0]-1))
        for i in range(M-1):
            Ls = forb[i][1] + 1
            Rs = forb[i+1][0] - 1
            if Ls <= Rs:
                safe.append((Ls, Rs))
        if forb[-1][1] < N:
            safe.append((forb[-1][1] + 1, N))
 
    # Utility: merge intervals (each given as a tuple (l, r)) into a union of disjoint intervals.
    def merge_intervals(intervals):
        if not intervals:
            return []
        intervals.sort(key=lambda x: x[0])
        merged = []
        cur_l, cur_r = intervals[0]
        for (l, r) in intervals[1:]:
            if l <= cur_r + 1:
                if r > cur_r:
                    cur_r = r
            else:
                merged.append((cur_l, cur_r))
                cur_l, cur_r = l, r
        merged.append((cur_l, cur_r))
        return merged
 
    # Utility: intersect a union of intervals (list 'union_list') with an interval 'intr'=(L,R)
    def intersect_intervals(union_list, intr):
        L, R = intr
        res = []
        for (l, r) in union_list:
            if r < L or l > R:
                continue
            nl = max(l, L)
            nr = min(r, R)
            if nl <= nr:
                res.append((nl, nr))
        return res
 
    # Utility: union two union-of-intervals lists.
    def union_of_intervals(list1, list2):
        return merge_intervals(list1 + list2)
     
    CAP = 100  # simulation window length (our DP “inside” one safe interval will look at at most CAP positions)
 
    # process_interval simulates, inside a safe interval "intr" = (L_int,R_int),
    # the DP (in normalized coordinates) for positions we can “land on” while staying in that safe interval.
    # "entry" is a list of normalized positions (with 0 corresponding to L_int).
    def process_interval(intr, entry, A, B, CAP):
        L_int, R_int = intr
        length = R_int - L_int + 1
        cap_val = min(length, CAP)
        dp = [False]*cap_val
        dq = deque()
        for v in entry:
            if 0 <= v < cap_val and not dp[v]:
                dp[v] = True
                dq.append(v)
        while dq:
            v = dq.popleft()
            for d in range(A, B+1):
                nxt = v + d
                if nxt < cap_val and not dp[nxt]:
                    dp[nxt] = True
                    dq.append(nxt)
        # Group consecutive True indices into intervals.
        intervals_dp = []
        i = 0
        while i < cap_val:
            if dp[i]:
                start = i
                j = i
                while j+1 < cap_val and dp[j+1]:
                    j += 1
                intervals_dp.append((start, j))
                i = j+1
            else:
                i += 1
        # If we simulated exactly CAP positions (and the safe interval is longer)
        # and if in some contiguous block the last simulated position (cap_val-1) is True,
        # then we “assume” the entire tail is reachable.
        full_tail = False
        tail_start = None
        if cap_val == CAP and length > CAP:
            for (s_i, e_i) in intervals_dp:
                if e_i == cap_val - 1:
                    full_tail = True
                    tail_start = s_i
                    break
        res = []
        for (s_i, e_i) in intervals_dp:
            if full_tail and e_i == cap_val - 1:
                res.append((L_int + s_i, R_int))
            else:
                res.append((L_int + s_i, L_int + e_i))
        return merge_intervals(res)
     
    # allowed_jump_set computes for a reached state R_state (a union of intervals in absolute coordinates)
    # the union of outcomes of a jump from any reached x (i.e. x+d with d in [A,B]).
    def allowed_jump_set(R_state, A, B):
        arr = []
        for (l, r) in R_state:
            arr.append((l + A, r + B))
        return merge_intervals(arr)
 
    # global_allowed will hold, as a union of intervals (in absolute coordinates),
    # all “landing positions” (even if not safe) that can be produced by a jump from some reached safe square.
    global_allowed = []
    reachedStates = {}  # dictionary: safe interval index -> reached state (union of intervals in that safe block)
 
    # Process safe intervals in increasing order.
    for i, intr in enumerate(safe):
        L_int, R_int = intr
        entry = []
        if i == 0:
            # The first safe interval must contain square 1.
            if L_int <= 1 <= R_int:
                entry = [1 - L_int]  # normalized coordinate (should be 0)
            else:
                continue
        else:
            # For later safe intervals, the available entry comes from global_allowed's intersection with the safe interval.
            inters = intersect_intervals(global_allowed, intr)
            if not inters:
                continue
            for (l_val, r_val) in inters:
                for x in range(l_val, r_val+1):
                    entry.append(x - L_int)
            entry = sorted(set(entry))
        if not entry:
            continue
        # Compute the reached state inside this safe interval.
        R_state = process_interval(intr, entry, A, B, CAP)
        reachedStates[i] = R_state	
        # If this safe interval contains N, test if N is reached.
        if L_int <= N <= R_int:
            for (l_val, r_val) in R_state:
                if l_val <= N <= r_val:
                    sys.stdout.write("Yes")
                    return
        # Compute the allowed jump outcomes from this reached state.
        V_i = allowed_jump_set(R_state, A, B)
        global_allowed = union_of_intervals(global_allowed, V_i)
 
    sys.stdout.write("No")
 
if __name__ == '__main__':
    main()