import bisect

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)  # 1-based indexing

    def update(self, idx, value):
        # idx is 1-based
        while idx <= self.n:
            if self.tree[idx] < value:
                self.tree[idx] = value
            else:
                break
            idx += idx & -idx

    def query(self, idx):
        # returns the maximum from 1..idx (1-based)
        res = 0
        while idx > 0:
            res = max(res, self.tree[idx])
            idx -= idx & -idx
        return res

def max_balanced_subsequence_sum(nums):
    if not nums:
        return 0
    s_list = [nums[i] - i for i in range(len(nums))]
    sorted_unique_s = sorted(list(set(s_list)))
    m = len(sorted_unique_s)
    fenwick = FenwickTree(m)
    max_sum = 0
    for s_i in s_list:
        r_i = bisect.bisect_left(sorted_unique_s, s_i)
        fenwick_idx = r_i + 1  # convert to 1-based
        max_prev = fenwick.query(fenwick_idx)
        current = s_i + max_prev
        # Update the Fenwick tree
        fenwick.update(fenwick_idx, current)
        if current > max_sum:
            max_sum = current
    return max_sum