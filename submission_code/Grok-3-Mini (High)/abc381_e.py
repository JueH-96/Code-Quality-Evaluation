import sys
import bisect

sys.setrecursionlimit(1000000)

class SegmentTree:
    def __init__(self, X_list, Y_list, start, end):
        self.start = start
        self.end = end
        if start == end:
            self.sorted_X = [X_list[start]]
            self.sorted_Y = [Y_list[start]]
            self.suff_min_Y = [Y_list[start]]
            self.left = None
            self.right = None
        else:
            mid = (start + end) // 2
            self.left = SegmentTree(X_list, Y_list, start, mid)
            self.right = SegmentTree(X_list, Y_list, mid + 1, end)
            left_X = self.left.sorted_X
            left_Y = self.left.sorted_Y
            right_X = self.right.sorted_X
            right_Y = self.right.sorted_Y
            merged_X = []
            merged_Y = []
            i, j = 0, 0
            len_left = len(left_X)
            len_right = len(right_X)
            while i < len_left and j < len_right:
                if left_X[i] <= right_X[j]:
                    merged_X.append(left_X[i])
                    merged_Y.append(left_Y[i])
                    i += 1
                else:
                    merged_X.append(right_X[j])
                    merged_Y.append(right_Y[j])
                    j += 1
            while i < len_left:
                merged_X.append(left_X[i])
                merged_Y.append(left_Y[i])
                i += 1
            while j < len_right:
                merged_X.append(right_X[j])
                merged_Y.append(right_Y[j])
                j += 1
            self.sorted_X = merged_X
            self.sorted_Y = merged_Y
            len_merged = len(merged_X)
            suff_min = [0] * len_merged
            suff_min[-1] = merged_Y[-1]
            for k in range(len_merged - 2, -1, -1):
                suff_min[k] = min(merged_Y[k], suff_min[k + 1])
            self.suff_min_Y = suff_min

    def has_point_in_range(self, ql, qr, val1, val2):
        return self._query(ql, qr, val1, val2)

    def _query(self, ql, qr, val1, val2):
        nl = self.start
        nr = self.end
        if ql > nr or qr < nl:
            return False
        if ql <= nl and nr <= qr:
            sorted_X_list = self.sorted_X
            idx = bisect.bisect_left(sorted_X_list, val1)
            if idx < len(sorted_X_list) and self.suff_min_Y[idx] <= val2:
                return True
            return False
        else:
            if self.left and self.left._query(ql, qr, val1, val2):
                return True
            if self.right and self.right._query(ql, qr, val1, val2):
                return True
            return False

data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
Q = int(data[index])
index += 1
S = data[index]
index += 1

cum1 = [0] * (N + 1)
cum2 = [0] * (N + 1)
for i in range(1, N + 1):
    cum1[i] = cum1[i - 1] + (1 if S[i - 1] == '1' else 0)
    cum2[i] = cum2[i - 1] + (1 if S[i - 1] == '2' else 0)

slash_pos = [p for p in range(1, N + 1) if S[p - 1] == '/']
M = len(slash_pos)

if M > 0:
    X_list = [cum1[p - 1] for p in slash_pos]
    Y_list = [cum2[p] for p in slash_pos]
    seg_tree = SegmentTree(X_list, Y_list, 0, M - 1)
else:
    seg_tree = None

for _ in range(Q):
    L = int(data[index])
    index += 1
    R = int(data[index])
    index += 1
    
    if M == 0 or seg_tree is None:
        print(0)
        continue
    
    left_idx = bisect.bisect_left(slash_pos, L)
    ub = bisect.bisect_right(slash_pos, R)
    if left_idx >= ub:
        print(0)
        continue
    
    slash_start_idx = left_idx
    slash_end_idx = ub - 1
    A_val = cum1[L - 1]
    B_val = cum2[R]
    
    low = 0
    high = 100000
    res_V = -1
    while low <= high:
        mid = (low + high) // 2
        val1 = mid + A_val
        val2 = B_val - mid
        if seg_tree.has_point_in_range(slash_start_idx, slash_end_idx, val1, val2):
            res_V = mid
            low = mid + 1
        else:
            high = mid - 1
    
    length = 2 * res_V + 1
    print(length)