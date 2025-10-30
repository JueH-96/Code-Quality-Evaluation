from typing import List

class Fenwick:
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

    def range_query(self, l, r):
        return self.query(r) - self.query(l - 1)

class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        if n <= 2:
            return [0] * len(queries)
        self.bit = Fenwick(n)
        # Initialize peaks in the BIT
        for j in range(1, n-1):
            if nums[j-1] < nums[j] and nums[j] > nums[j+1]:
                self.bit.update(j + 1, 1)
        res = []
        for q in queries:
            if q[0] == 1:
                l, r = q[1], q[2]
                left = l + 1
                right = r - 1
                if left > right:
                    res.append(0)
                else:
                    ans = self.bit.range_query(left + 1, right + 1)
                    res.append(ans)
            else:
                # Update query
                i, val = q[1], q[2]
                positions = [i-1, i, i+1]
                # Remove old peaks
                for j in positions:
                    if 0 < j < n - 1:
                        if nums[j-1] < nums[j] and nums[j] > nums[j+1]:
                            self.bit.update(j + 1, -1)
                # Update nums[i]
                nums[i] = val
                # Add new peaks
                for j in positions:
                    if 0 < j < n - 1:
                        if nums[j-1] < nums[j] and nums[j] > nums[j+1]:
                            self.bit.update(j + 1, 1)
        return res