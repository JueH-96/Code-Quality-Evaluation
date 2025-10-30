import sys

def solve():
    """
    Solves the problem by considering cases on the number of type-2 operations
    and using binary lifting for fast interval covering cost calculation.
    """
    sys.setrecursionlimit(2 * 10**6)

    def get_ints():
        return list(map(int, sys.stdin.readline().split()))

    N, M = get_ints()
    if M == 0 and N > 0:
        print(-1)
        return
    ops_orig = [get_ints() for _ in range(M)]
    ops = [(l - 1, r - 1) for l, r in ops_orig]

    # --- Precomputation for cost_1(L, R) ---
    max_r_at = [-1] * N
    op_idx_at = [-1] * N
    for i, (l, r) in enumerate(ops):
        if r > max_r_at[l]:
            max_r_at[l] = r
            op_idx_at[l] = i

    up_val = [-1] * N
    up_op = [-1] * N
    
    # up_val[i] = max reach from an interval starting at or before i
    # up_op[i] = the op giving that reach
    current_max_r = -1
    current_op_idx = -1
    for i in range(N):
        if max_r_at[i] > current_max_r:
            current_max_r = max_r_at[i]
            current_op_idx = op_idx_at[i]
        up_val[i] = current_max_r
        up_op[i] = current_op_idx

    LOGN = (N).bit_length()
    
    # up_p[k][i] = position after 2^k jumps, starting from covering up to i-1
    up_p = [[i for i in range(N)] for _ in range(LOGN)]
    for i in range(N):
        if up_val[i] > i:
            up_p[0][i] = up_val[i]
    
    for k in range(LOGN - 1):
        for i in range(N):
            if up_p[k][i] < N:
                up_p[k+1][i] = up_p[k][up_p[k][i]]
            else:
                up_p[k+1][i] = up_p[k][i]

    def get_cost_1_fast(start, end):
        if start > end: return 0
        if up_val[start] < start: return float('inf')

        cost = 1
        pos = start
        
        for k in range(LOGN - 1, -1, -1):
            if up_p[k][pos] < end:
                pos = up_p[k][pos]
                cost += (1 << k)
        
        if pos < end:
            pos = up_p[0][pos]
        
        return cost if pos >= end else float('inf')

    def reconstruct_1(start, end):
        if start > end: return []
        ops_path = []
        curr = start
        while curr <= end:
            op_to_use = up_op[curr]
            if op_to_use == -1 or ops[op_to_use][1] < curr: return None
            
            ops_path.append(op_to_use)
            next_curr = ops[op_to_use][1] + 1
            if next_curr <= curr: return None
            curr = next_curr
        return ops_path

    min_cost = float('inf')
    best_op_choices = []

    # Case 0: Only type 1
    cost = get_cost_1_fast(0, N - 1)
    if cost < min_cost:
        min_cost = cost
        path = reconstruct_1(0, N - 1)
        ans = [0] * M
        for op_idx in path:
            ans[op_idx] = 1
        best_op_choices = ans

    # Case 1: One type 2
    for i in range(M):
        l, r = ops[i]
        cost = get_cost_1_fast(l, r)
        if cost != float('inf'):
            total_cost = 1 + cost
            if total_cost < min_cost:
                min_cost = total_cost
                path = reconstruct_1(l, r)
                ans = [0] * M
                ans[i] = 2
                for op_idx in path:
                    ans[op_idx] = 1
                best_op_choices = ans

    # Case 2: Two type 2
    unique_L = sorted(list(set(l for l, r in ops)))
    unique_R = sorted(list(set(r for l, r in ops)))
    
    op_by_L = {l: i for i, (l, r) in reversed(list(enumerate(ops)))}
    op_by_R = {r: i for i, (l, r) in reversed(list(enumerate(ops)))}

    r_ptr = 0
    for l_val in unique_L:
        while r_ptr < len(unique_R) and unique_R[r_ptr] < l_val:
            r_ptr += 1
        if r_ptr < len(unique_R):
            r_val = unique_R[r_ptr]
            cost = get_cost_1_fast(l_val, r_val)
            if cost != float('inf'):
                total_cost = 2 + cost
                if total_cost < min_cost:
                    min_cost = total_cost
                    path = reconstruct_1(l_val, r_val)
                    ans = [0] * M
                    op1_idx = op_by_L[l_val]
                    op2_idx = op_by_R[r_val]
                    ans[op1_idx] = 2
                    ans[op2_idx] = 2
                    for op_idx in path:
                        ans[op_idx] = 1
                    best_op_choices = ans
    
    # Case 3: Full cover by type 2
    sorted_ops_by_L = sorted(enumerate(ops), key=lambda x: x[1][0])
    min_r_so_far = float('inf')
    op_for_min_r = -1
    
    for i, (l, r) in sorted_ops_by_L:
        if l > min_r_so_far:
            if 2 < min_cost:
                min_cost = 2
                ans = [0] * M
                ans[i] = 2
                ans[op_for_min_r] = 2
                best_op_choices = ans
            break
        if r < min_r_so_far:
             min_r_so_far = r
             op_for_min_r = i

    if min_cost == float('inf'):
        print(-1)
    else:
        print(min_cost)
        print(*best_op_choices)

if __name__ == "__main__":
    solve()