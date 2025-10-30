import sys
INF = 10**15

def dist_cycle(S, T, forbid, M):
    if T in forbid or S in forbid:
        return -1
    D_cw = (T - S) % M
    D_ccw = (S - T) % M
    cw_usable = True
    for F in forbid:
        if 0 <= (F - S) % M <= D_cw:
            cw_usable = False
            break
    ccw_usable = True
    for F in forbid:
        if 0 <= (S - F) % M <= D_ccw:
            ccw_usable = False
            break
    if cw_usable and ccw_usable:
        return min(D_cw, D_ccw)
    elif cw_usable:
        return D_cw
    elif ccw_usable:
        return D_ccw
    else:
        return -1

def cost_node(b, P_left_opt, P_right_opt, L, R, A, B, M, N, K):
    idx = L + b
    forbid_list = []
    # F_left
    if idx > 0:  # left neighbor exists
        if b > 0:  # left in S
            if P_left_opt:  # P_left_opt true, i-1 before i
                F_left = B[idx - 1]
            else:  # P_left_opt false
                F_left = A[idx - 1]
        else:  # b == 0, left not in S
            F_left = A[idx - 1]
        forbid_list.append(F_left)
    # F_right
    if idx < N - 1:  # right neighbor exists
        if b < K - 1:  # right in S
            if P_right_opt:  # P_right_opt true, i+1 before i
                F_right = B[idx + 1]
            else:  # P_right_opt false
                F_right = A[idx + 1]
        else:  # b == K - 1, right not in S
            F_right = A[idx + 1]
        forbid_list.append(F_right)
    dist = dist_cycle(A[idx], B[idx], forbid_list, M)
    if dist == -1:
        return INF
    else:
        return dist

def compute_block_min_sum(L, R, K, A, B, M, N):
    if K == 1:
        idx = L
        forbid_list = []
        if idx > 0:
            forbid_list.append(A[idx - 1])
        if idx < N - 1:
            forbid_list.append(A[idx + 1])
        dist = dist_cycle(A[idx], B[idx], forbid_list, M)
        if dist == -1:
            return INF
        else:
            return dist
    else:  # K >= 2
        # compute for b=0
        cost_b0_p0 = cost_node(0, None, 0, L, R, A, B, M, N, K)
        cost_b0_p1 = cost_node(0, None, 1, L, R, A, B, M, N, K)
        if K == 2:
            sum_p0 = cost_b0_p0 + cost_node(1, 1 - 0, None, L, R, A, B, M, N, K)
            sum_p1 = cost_b0_p1 + cost_node(1, 1 - 1, None, L, R, A, B, M, N, K)
            min_sum = min(sum_p0, sum_p1)
            return min_sum
        else:  # K > 2
            prev_dp = [cost_b0_p0, cost_b0_p1]
            for b in range(1, K - 1):  # b from 1 to K-2
                curr_dp_val0 = INF
                curr_dp_val1 = INF
                for p in range(2):  # P_right for b
                    for p_prev in range(2):  # P_right for b-1
                        P_left_b_val = 1 - p_prev
                        cost_b_val = cost_node(b, P_left_b_val, p, L, R, A, B, M, N, K)
                        if prev_dp[p_prev] < INF and cost_b_val < INF:
                            sum_val = prev_dp[p_prev] + cost_b_val
                            if p == 0 and sum_val < curr_dp_val0:
                                curr_dp_val0 = sum_val
                            elif p == 1 and sum_val < curr_dp_val1:
                                curr_dp_val1 = sum_val
                prev_dp = [curr_dp_val0, curr_dp_val1]
            # now prev_dp is for b=K-2, sum up to K-2
            min_sum = INF
            for p in range(2):
                if prev_dp[p] < INF:
                    P_left_K1_val = 1 - p
                    cost_K1_val = cost_node(K - 1, P_left_K1_val, None, L, R, A, B, M, N, K)
                    sum_val = prev_dp[p] + cost_K1_val
                    if sum_val < min_sum:
                        min_sum = sum_val
            return min_sum

# main code
data = sys.stdin.read().split()
index = 0
N = int(data[index])
M = int(data[index + 1])
index += 2
A = list(map(int, data[index:index + N]))
index += N
B = list(map(int, data[index:index + N]))

ans = 0
impossible_flag = False
i = 0
while i < N:
    if A[i] != B[i]:
        L = i
        j = i
        while j < N and A[j] != B[j]:
            j += 1
        R = j - 1  # inclusive
        K = R - L + 1
        min_sum_block = compute_block_min_sum(L, R, K, A, B, M, N)
        if min_sum_block >= INF:
            impossible_flag = True
            break
        else:
            ans += min_sum_block
        i = j  # move to next after block
    else:
        i += 1

if impossible_flag:
    print(-1)
else:
    print(ans)