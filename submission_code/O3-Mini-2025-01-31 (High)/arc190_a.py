# YOUR CODE HERE
def main():
    import sys,sys, bisect
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    try:
        N = int(next(it))
        M = int(next(it))
    except:
        return

    ops = []  # Each op: (L, R, index)
    for i in range(M):
        try:
            L = int(next(it)); R = int(next(it))
        except:
            break
        ops.append((L, R, i))
        
    # Special case: if M==1, then only possibility is op1 with [1,N].
    if M == 1:
        L, R, idx = ops[0]
        if L == 1 and R == N:
            sys.stdout.write("1
1")
        else:
            sys.stdout.write("-1")
        return

    # Candidate A: Greedy op1 only cover.
    def greedy_op1_cover(ops_list, N):
        sorted_ops = sorted(ops_list, key=lambda x: x[0])
        res = []
        current = 1
        i = 0
        Ls = len(sorted_ops)
        while current <= N:
            best_R = current - 1
            best_op = None
            while i < Ls and sorted_ops[i][0] <= current:
                if sorted_ops[i][1] > best_R:
                    best_R = sorted_ops[i][1]
                    best_op = sorted_ops[i][2]
                i += 1
            if best_op is None:
                return None
            res.append(best_op)
            current = best_R + 1
        return res

    candidate_A = greedy_op1_cover(ops, N)
    if candidate_A is not None and len(candidate_A) == 1:
        ans = ["0"] * M
        ans[candidate_A[0]] = "1"
        sys.stdout.write("1
" + " ".join(ans))
        return

    # Candidate B: try two op2's with disjoint holes.
    sorted_by_L = sorted(ops, key=lambda x: x[0])
    candidateB_pair = None
    min_R = 10**10
    min_R_op = None
    for (L, R, idx) in sorted_by_L:
        if min_R_op is not None:
            if L > min_R:
                candidateB_pair = (min_R_op, idx)
                break
        if R < min_R:
            min_R = R
            min_R_op = idx
    if candidateB_pair is not None:
        res_assignment = ["0"] * M
        a, b = candidateB_pair
        res_assignment[a] = "2"
        res_assignment[b] = "2"
        sys.stdout.write("2
" + " ".join(res_assignment))
        return

    # Candidate C: Find a pair (op_i, op_j) with op_i as op1 and op_j as op2 such that [L_j,R_j] ⊆ [L_i,R_i].
    L_arr = [x[0] for x in sorted_by_L]
    R_arr = [x[1] for x in sorted_by_L]
    idx_arr = [x[2] for x in sorted_by_L]
    n_ops = len(sorted_by_L)
    prefix_max_R = [0] * n_ops
    prefix_max_idx = [0] * n_ops
    prefix_second_max_R = [ -1 ] * n_ops
    prefix_second_max_idx = [ -1 ] * n_ops
    prefix_max_R[0] = R_arr[0]
    prefix_max_idx[0] = idx_arr[0]
    prefix_second_max_R[0] = -1
    prefix_second_max_idx[0] = -1
    for i in range(1, n_ops):
        if R_arr[i] > prefix_max_R[i-1]:
            prefix_second_max_R[i] = prefix_max_R[i-1]
            prefix_second_max_idx[i] = prefix_max_idx[i-1]
            prefix_max_R[i] = R_arr[i]
            prefix_max_idx[i] = idx_arr[i]
        else:
            prefix_max_R[i] = prefix_max_R[i-1]
            prefix_max_idx[i] = prefix_max_idx[i-1]
            if R_arr[i] > prefix_second_max_R[i-1]:
                prefix_second_max_R[i] = R_arr[i]
                prefix_second_max_idx[i] = idx_arr[i]
            else:
                prefix_second_max_R[i] = prefix_second_max_R[i-1]
                prefix_second_max_idx[i] = prefix_second_max_idx[i-1]
    candidateC_pair = None
    for (L_cur, R_cur, idx_cur) in ops:
        pos = bisect.bisect_right(L_arr, L_cur)
        if pos <= 0:
            continue
        p = pos - 1
        cand_op = prefix_max_idx[p]
        cand_R = prefix_max_R[p]
        chosen = None
        if cand_op == idx_cur:
            if prefix_second_max_R[p] >= R_cur:
                chosen = prefix_second_max_idx[p]
        else:
            if cand_R >= R_cur:
                chosen = cand_op
        if chosen is not None:
            candidateC_pair = (chosen, idx_cur)
            break
    if candidateC_pair is not None:
        res_assignment = ["0"] * M
        a, b = candidateC_pair  # a used as op1, b as op2.
        res_assignment[a] = "1"
        res_assignment[b] = "2"
        sys.stdout.write("2
" + " ".join(res_assignment))
        return

    # Otherwise, if candidate A exists, output it.
    if candidate_A is not None:
        cost = len(candidate_A)
        res_assignment = ["0"] * M
        for op_idx in candidate_A:
            res_assignment[op_idx] = "1"
        sys.stdout.write(str(cost) + "
" + " ".join(res_assignment))
    else:
        sys.stdout.write("-1")
        
if __name__ == '__main__':
    main()