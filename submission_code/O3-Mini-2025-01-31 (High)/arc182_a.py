def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    try:
        N = int(next(it))
        Q = int(next(it))
    except StopIteration:
        return
    # Read Q operations as (p,v)
    ops = []
    for _ in range(Q):
        p = int(next(it))
        v = int(next(it))
        ops.append((p, v))
        
    mod = 998244353
    # DP state:
    # Each state is represented as a tuple (L, R, a, b) with 0 <= L < R <= N+1.
    # Interpretation:
    #   S[1..L] are updated by some left–ops to value a.
    #   S[R..N] are updated by some right–ops to value b.
    #   The indices L+1 .. R–1 are still 0.
    # For a uniform state (i.e. the whole array S is updated to v) we store it as (N, N+1, v, v).
    # Initially S is all 0 so state = (0, N+1, 0, 0)
    dp = {}
    dp[(0, N+1, 0, 0)] = 1
    
    # Helper routine to add count for a state.
    def add_state(state, cnt, d):
        if cnt == 0:
            return
        d[state] = (d.get(state, 0) + cnt) % mod
        
    # Process operations one by one.
    for (p_op, v_op) in ops:
        newdp = {}
        for (L, R, a, b), cnt in dp.items():
            # Distinguish uniform state from non-uniform.
            if L == N and R == N+1 and a == b:
                u = a
                # Option 1: left update on uniform state.
                if u <= v_op:
                    if p_op < N:
                        add_state((p_op, p_op+1, v_op, u), cnt, newdp)
                    elif p_op == N:
                        add_state((N, N+1, v_op, v_op), cnt, newdp)
                # Option 2: right update on uniform state.
                if u <= v_op:
                    if p_op > 1:
                        add_state((p_op-1, p_op, u, v_op), cnt, newdp)
                    elif p_op == 1:
                        add_state((N, N+1, v_op, v_op), cnt, newdp)
            else:
                # Non-uniform state.
                # Option 1: Choose left update.
                # Case (a): if p_op < R (non–crossing left update).
                if p_op < R:
                    if L == 0 or a <= v_op:
                        newL = L
                        if p_op > L:
                            newL = p_op
                        if newL < R:
                            add_state((newL, R, v_op, b), cnt, newdp)
                # Case (b): if p_op >= R (crossing left update).
                if p_op >= R:
                    if max(a, b) <= v_op:
                        if p_op < N:
                            add_state((p_op, p_op+1, v_op, b), cnt, newdp)
                        elif p_op == N:
                            add_state((N, N+1, v_op, v_op), cnt, newdp)
                # Option 2: Choose right update.
                # Case (a): if p_op > L (non–crossing right update).
                if p_op > L:
                    if R == N+1 or b <= v_op:
                        newR = p_op
                        if L < newR:
                            add_state((L, newR, a, v_op), cnt, newdp)
                # Case (b): if p_op <= L (crossing right update).
                if p_op <= L:
                    if max(a, b) <= v_op:
                        if p_op > 1:
                            add_state((p_op-1, p_op, a, v_op), cnt, newdp)
                        elif p_op == 1:
                            add_state((N, N+1, v_op, v_op), cnt, newdp)
        dp = newdp
        if not dp:
            print(0)
            return
    ans = sum(dp.values()) % mod
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()