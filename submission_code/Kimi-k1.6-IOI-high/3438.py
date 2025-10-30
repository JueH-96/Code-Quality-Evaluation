class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)  # 1-based indexing

    def update(self, idx, delta):
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & -idx

    def query(self, idx):
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

class Solution:
    def countOfPeaks(self, nums: list, queries: list) -> list:
        n = len(nums)
        peaks = [0] * n
        # Precompute initial peaks
        for j in range(n):
            if j == 0 or j == n - 1:
                continue
            if nums[j] > nums[j - 1] and nums[j] > nums[j + 1]:
                peaks[j] = 1
        # Initialize Fenwick Tree
        fenwick = FenwickTree(n)
        for j in range(n):
            if peaks[j] == 1:
                fenwick.update(j + 1, 1)  # Convert to 1-based index
        # Process each query
        result = []
        for q in queries:
            if q[0] == 1:
                # Type 1: count peaks in [l, r]
                l, r = q[1], q[2]
                if l >= r - 1:
                    result.append(0)
                else:
                    a = l + 1
                    b = r - 1
                    sum_peaks = fenwick.query(b + 1) - fenwick.query(a)
                    result.append(sum_peaks)
            else:
                # Type 2: update nums[index_i] to val_i
                idx = q[1]
                new_val = q[2]
                old_val = nums[idx]
                nums[idx] = new_val
                # Check positions idx-1, idx, idx+1
                for dj in (-1, 0, 1):
                    j = idx + dj
                    if j < 0 or j >= n:
                        continue
                    # Determine new peak status
                    new_peak = 0
                    if 0 < j < n - 1 and nums[j] > nums[j - 1] and nums[j] > nums[j + 1]:
                        new_peak = 1
                    if peaks[j] != new_peak:
                        delta = new_peak - peaks[j]
                        peaks[j] = new_peak
                        fenwick.update(j + 1, delta)  # Convert to 1-based index
        return result