import sys
import bisect

sys.setrecursionlimit(300000)

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    it = iter(data)
    n = int(next(it))
    weights = [int(next(it)) for _ in range(n)]
    intervals = []
    for _ in range(n):
        L_i = int(next(it))
        R_i = int(next(it))
        intervals.append((L_i, R_i))
    
    min_R = min(R_i for L_i, R_i in intervals)
    max_L = max(L_i for L_i, R_i in intervals)
    
    universal = [False] * n
    for i in range(n):
        L_i, R_i = intervals[i]
        if L_i <= min_R and R_i >= max_L:
            universal[i] = True
            
    big_comp_exists = (min_R < max_L)
    comp_id = [0] * n
    if big_comp_exists:
        for i in range(n):
            if not universal[i]:
                comp_id[i] = 1
                
    q = int(next(it))
    queries = []
    for _ in range(q):
        s = int(next(it)) - 1
        t = int(next(it)) - 1
        queries.append((s, t))
        
    if big_comp_exists:
        M = [i for i in range(n) if comp_id[i] == 1]
        if M:
            arr1 = sorted([(intervals[i][1], weights[i]) for i in M])
            prefix_min = [0] * len(arr1)
            prefix_min[0] = arr1[0][1]
            for i in range(1, len(arr1)):
                prefix_min[i] = min(prefix_min[i-1], arr1[i][1])
                
            def query_struct1(T):
                low, high = 0, len(arr1) - 1
                idx = -1
                while low <= high:
                    mid = (low + high) // 2
                    if arr1[mid][0] < T:
                        idx = mid
                        low = mid + 1
                    else:
                        high = mid - 1
                if idx == -1:
                    return 10**18
                return prefix_min[idx]
                    
            arr2 = sorted([(intervals[i][0], weights[i]) for i in M])
            suffix_min = [0] * len(arr2)
            if arr2:
                suffix_min[-1] = arr2[-1][1]
                for i in range(len(arr2)-2, -1, -1):
                    suffix_min[i] = min(suffix_min[i+1], arr2[i][1])
                    
            def query_struct2(T):
                low, high = 0, len(arr2) - 1
                idx = len(arr2)
                while low <= high:
                    mid = (low + high) // 2
                    if arr2[mid][0] > T:
                        idx = mid
                        high = mid - 1
                    else:
                        low = mid + 1
                if idx == len(arr2):
                    return 10**18
                return suffix_min[idx]
                
            all_L_set = set(L for L, R in intervals) | set(intervals[i][0] for i in M)
            all_L = sorted(all_L_set)
            n_seg = len(all_L)
            
            class SegmentTree:
                def __init__(self, size):
                    self.n = size
                    self.tree = [10**18] * (4 * size)
                
                def update(self, idx, l, r, pos, val):
                    if l == r:
                        self.tree[idx] = min(self.tree[idx], val)
                        return
                    mid = (l + r) // 2
                    if pos <= mid:
                        self.update(idx*2, l, mid, pos, val)
                    else:
                        self.update(idx*2+1, mid+1, r, pos, val)
                    self.tree[idx] = min(self.tree[idx*2], self.tree[idx*2+1])
                
                def query(self, idx, l, r, ql, qr):
                    if ql > qr:
                        return 10**18
                    if ql <= l and r <= qr:
                        return self.tree[idx]
                    mid = (l + r) // 2
                    left_res = 10**18
                    right_res = 10**18
                    if ql <= mid:
                        left_res = self.query(idx*2, l, mid, ql, qr)
                    if qr > mid:
                        right_res = self.query(idx*2+1, mid+1, r, ql, qr)
                    return min(left_res, right_res)
            
            st = SegmentTree(n_seg)
            events_case2 = []
            for i in M:
                L_i, R_i = intervals[i]
                W_i = weights[i]
                events_case2.append((R_i, 1, L_i, W_i))
            for idx_q, (s0, t0) in enumerate(queries):
                if comp_id[s0] == 1 and comp_id[t0] == 1:
                    L_s = intervals[s0][0]
                    R_t = intervals[t0][1]
                    if R_t < L_s:
                        events_case2.append((L_s, 2, R_t, idx_q))
            events_case2.sort(key=lambda x: (x[0], x[1]))
            
            res_case2 = [10**18] * q
            for event in events_case2:
                coord = event[0]
                typ = event[1]
                if typ == 1:
                    L_x = event[2]
                    W_x = event[3]
                    pos = bisect.bisect_left(all_L, L_x)
                    st.update(1, 0, n_seg-1, pos, W_x)
                else:
                    R_t_val = event[2]
                    idx_q = event[3]
                    ptr = bisect.bisect_right(all_L, R_t_val)
                    if ptr >= n_seg:
                        res = 10**18
                    else:
                        res = st.query(1, 0, n_seg-1, ptr, n_seg-1)
                    res_case2[idx_q] = res
            
            events_case3 = []
            for i in M:
                L_i, R_i = intervals[i]
                W_i = weights[i]
                events_case3.append((R_i, 1, L_i, W_i))
            for idx_q, (s0, t0) in enumerate(queries):
                if comp_id[s0] == 1 and comp_id[t0] == 1:
                    R_s = intervals[s0][1]
                    L_t = intervals[t0][0]
                    if R_s < L_t:
                        events_case3.append((L_t, 2, R_s, idx_q))
            events_case3.sort(key=lambda x: (x[0], x[1]))
            
            st2 = SegmentTree(n_seg)
            res_case3 = [10**18] * q
            for event in events_case3:
                coord = event[0]
                typ = event[1]
                if typ == 1:
                    L_x = event[2]
                    W_x = event[3]
                    pos = bisect.bisect_left(all_L, L_x)
                    st2.update(1, 0, n_seg-1, pos, W_x)
                else:
                    R_s_val = event[2]
                    idx_q = event[3]
                    ptr = bisect.bisect_right(all_L, R_s_val)
                    if ptr >= n_seg:
                        res = 10**18
                    else:
                        res = st2.query(1, 0, n_seg-1, ptr, n_seg-1)
                    res_case3[idx_q] = res
        else:
            res_case2 = [10**18] * q
            res_case3 = [10**18] * q
    else:
        res_case2 = [10**18] * q
        res_case3 = [10**18] * q
        
    output_lines = []
    for idx_q, (s, t) in enumerate(queries):
        if comp_id[s] != comp_id[t] or comp_id[s] == 0:
            output_lines.append("-1")
        else:
            L_s, R_s = intervals[s]
            L_t, R_t = intervals[t]
            if R_s < L_t or R_t < L_s:
                candidate0 = weights[s] + weights[t]
            else:
                candidate0 = 10**18
                
            if big_comp_exists and M:
                case1 = query_struct1(min(L_s, L_t))
                case4 = query_struct2(max(R_s, R_t))
                case2_val = res_case2[idx_q]
                case3_val = res_case3[idx_q]
                min_adj_both = min(case1, case2_val, case3_val, case4)
                candidate1 = weights[s] + weights[t] + min_adj_both
                ans = min(candidate0, candidate1)
                if ans >= 10**18:
                    ans = -1
                output_lines.append(str(ans))
            else:
                ans = candidate0 if candidate0 < 10**18 else -1
                output_lines.append(str(ans))
                
    print("
".join(output_lines))

if __name__ == "__main__":
    main()