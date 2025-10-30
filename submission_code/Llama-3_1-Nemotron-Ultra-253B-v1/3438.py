from typing import List

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)  # 1-based indexing

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

    def range_query(self, l, r):
        if l > r:
            return 0
        return self.query(r) - self.query(l - 1)

class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        is_peak = [False] * n
        for i in range(n):
            if i == 0 or i == n - 1:
                is_peak[i] = False
            else:
                is_peak[i] = (nums[i] > nums[i-1] and nums[i] > nums[i+1])
        
        fenwick = FenwickTree(n)
        for i in range(n):
            if is_peak[i]:
                fenwick.update(i + 1, 1)
        
        answer = []
        for q in queries:
            if q[0] == 1:
                l, r = q[1], q[2]
                start = l + 1
                end = r - 1
                if start > end:
                    answer.append(0)
                else:
                    cnt = fenwick.range_query(start + 1, end + 1)
                    answer.append(cnt)
            else:
                index, val = q[1], q[2]
                if nums[index] == val:
                    continue
                nums[index] = val
                for j in [index - 1, index, index + 1]:
                    if j < 0 or j >= n:
                        continue
                    if j == 0 or j == n - 1:
                        if is_peak[j]:
                            fenwick.update(j + 1, -1)
                            is_peak[j] = False
                        continue
                    prev_peak = is_peak[j]
                    current_peak = (nums[j] > nums[j-1] and nums[j] > nums[j+1])
                    if prev_peak != current_peak:
                        delta = 1 if current_peak else -1
                        fenwick.update(j + 1, delta)
                        is_peak[j] = current_peak
        return answer