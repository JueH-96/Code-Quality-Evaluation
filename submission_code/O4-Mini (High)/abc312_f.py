import sys
import threading
def main():
    import sys
    data = sys.stdin
    # read N, M
    line = data.readline().split()
    if not line:
        return
    N = int(line[0]); M = int(line[1])
    P = []  # pull-tab cans (type 0)
    R = []  # regular cans (type 1)
    O = []  # opener capacities (type 2)
    for _ in range(N):
        t_x = data.readline().split()
        if not t_x:
            t_x = data.readline().split()
            if not t_x:
                break
        t = int(t_x[0]); x = int(t_x[1])
        if t == 0:
            P.append(x)
        elif t == 1:
            R.append(x)
        else:
            O.append(x)
    # sort descending
    P.sort(reverse=True)
    R.sort(reverse=True)
    O.sort(reverse=True)
    import heapq
    # initial selection of pull-tabs
    # we maintain a min-heap S_big of selected items' values,
    # always holding the top L_prev values from the pool
    L_prev = M
    S_big = []
    sum_S_big = 0
    p_len = len(P)
    # take up to M pull-tabs initially
    if p_len > 0:
        take0 = p_len if p_len < L_prev else L_prev
        # copy the top take0 pulls
        S_big = P[:take0]
        heapq.heapify(S_big)
        sum_S_big = sum(S_big)
    # answer for k2 = 0 (no openers)
    ans = sum_S_big
    # we'll integrate regs up to capacity, tracked by r_idx
    r_idx = 0
    r_len = len(R)
    o_len = len(O)
    # maximum number of openers we can pick is min(o_len, M)
    k2_max = o_len if o_len < M else M
    # iterate choosing k2 openers one by one
    for i in range(k2_max):
        # opener capacity value
        cap = O[i]
        # new total capacity in terms of how many regs we may now open
        # cannot exceed number of regs available
        new_total = r_idx + cap
        if new_total > r_len:
            new_total = r_len
        # integrate the newly allowed regular cans (their values R[r_idx:new_total])
        # into our pool, updating the selected top-L_prev items
        while r_idx < new_total:
            v = R[r_idx]
            r_idx += 1
            # if we have room in the selection (size < L_prev), just add
            if len(S_big) < L_prev:
                heapq.heappush(S_big, v)
                sum_S_big += v
            else:
                # selection is full of size L_prev >= 1
                # if this reg is better than the smallest selected, swap in
                # heapreplace pops the smallest and pushes v
                if S_big and S_big[0] < v:
                    t = heapq.heapreplace(S_big, v)
                    sum_S_big += v - t
                # otherwise skip v (it stays out of selection)
        # now we've processed all new regs; update capacity pointer
        # reduce selection size by 1 because we used one item slot for the opener
        L_cur = L_prev - 1
        # if our selection heap is too big, pop the smallest
        if len(S_big) > L_cur:
            t = heapq.heappop(S_big)
            sum_S_big -= t
        # update for next iteration
        L_prev = L_cur
        # update answer
        if sum_S_big > ans:
            ans = sum_S_big
    # output the result
    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()