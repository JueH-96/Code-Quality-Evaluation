from typing import List

class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        class FenwickTree:
            def __init__(self, size):
                self.n = size
                self.tree = [0] * (self.n + 1)  # 1-based indexing

            def update(self, index, delta):
                while index <= self.n:
                    self.tree[index] += delta
                    index += index & -index

            def query(self, index):
                res = 0
                while index > 0:
                    res += self.tree[index]
                    index -= index & -index
                return res

        n = len(nums)
        fenwick = FenwickTree(n)

        # Initialize the Fenwick Tree with peaks from the initial array
        for j in range(n):
            if j == 0 or j == n - 1:
                continue
            if nums[j] > nums[j - 1] and nums[j] > nums[j + 1]:
                fenwick.update(j + 1, 1)

        answer = []
        for q in queries:
            if q[0] == 1:
                l, r = q[1], q[2]
                res = fenwick.query(r) - fenwick.query(l + 1)
                answer.append(res)
            else:
                index_i, val_i = q[1], q[2]
                # Check and update peaks around index_i before the change
                for j in [index_i - 1, index_i, index_i + 1]:
                    if 0 <= j < n:
                        if 1 <= j <= n - 2:
                            if nums[j] > nums[j - 1] and nums[j] > nums[j + 1]:
                                fenwick.update(j + 1, -1)
                # Update the value in nums
                nums[index_i] = val_i
                # Check and update peaks around index_i after the change
                for j in [index_i - 1, index_i, index_i + 1]:
                    if 0 <= j < n:
                        if 1 <= j <= n - 2:
                            if nums[j] > nums[j - 1] and nums[j] > nums[j + 1]:
                                fenwick.update(j + 1, 1)
        return answer