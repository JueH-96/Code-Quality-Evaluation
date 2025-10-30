class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)  # 1-based indexing

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

    def range_query(self, l, r):
        if l > r:
            return 0
        return self.query(r) - self.query(l - 1)

class Solution:
    def countOfPeaks(self, nums, queries):
        n = len(nums)
        ft = FenwickTree(n)
        peaks = [False] * n

        # Precompute initial peaks
        for i in range(1, n - 1):
            if nums[i-1] < nums[i] and nums[i] > nums[i+1]:
                peaks[i] = True
                ft.update(i + 1, 1)  # BIT is 1-based

        answer = []
        for q in queries:
            if q[0] == 1:
                l, r = q[1], q[2]
                a = l + 1
                b = r - 1
                if a > b:
                    answer.append(0)
                else:
                    res = ft.range_query(a + 1, b + 1)
                    answer.append(res)
            else:
                index, val = q[1], q[2]
                if nums[index] == val:
                    continue
                # Collect affected positions
                affected = []
                for delta in (-1, 0, 1):
                    pos = index + delta
                    if 0 <= pos < n:
                        affected.append(pos)
                # Compute pre status
                pre = {}
                for pos in affected:
                    if 1 <= pos <= n - 2:
                        pre_peak = (nums[pos-1] < nums[pos] and nums[pos] > nums[pos+1])
                    else:
                        pre_peak = False
                    pre[pos] = pre_peak
                # Update the array
                nums[index] = val
                # Compute post status
                post = {}
                for pos in affected:
                    if 1 <= pos <= n - 2:
                        post_peak = (nums[pos-1] < nums[pos] and nums[pos] > nums[pos+1])
                    else:
                        post_peak = False
                    post[pos] = post_peak
                # Update Fenwick Tree based on pre and post
                for pos in affected:
                    if 1 <= pos <= n - 2:
                        if pre[pos] != post[pos]:
                            delta_bit = 1 if post[pos] else -1
                            ft.update(pos + 1, delta_bit)
                            peaks[pos] = post[pos]
        return answer