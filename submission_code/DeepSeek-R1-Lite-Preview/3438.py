from typing import List

class FenwickTree:
    def __init__(self, size):
        self.N = size
        self.tree = [0] * (self.N + 1)
    
    def update(self, index, delta):
        while index <= self.N:
            self.tree[index] += delta
            index += index & -index
    
    def query(self, index):
        res = 0
        while index > 0:
            res += self.tree[index]
            index -= index & -index
        return res
    
    def query_range(self, left, right):
        return self.query(right) - self.query(left - 1)

class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        N = len(nums)
        peaks = [0] * N
        for i in range(1, N - 1):
            if nums[i] > nums[i - 1] and nums[i] > nums[i + 1]:
                peaks[i] = 1
        fenwick = FenwickTree(N)
        for i in range(1, N - 1):
            if peaks[i]:
                fenwick.update(i, 1)
        answer = []
        for query in queries:
            if query[0] == 1:
                l = query[1]
                r = query[2]
                left = max(1, l + 1)
                right = min(N - 2, r - 1)
                if left > right:
                    answer.append(0)
                else:
                    answer.append(fenwick.query_range(left, right))
            elif query[0] == 2:
                index = query[1]
                val = query[2]
                old_val = nums[index]
                nums[index] = val
                for i in [index - 1, index, index + 1]:
                    if 1 <= i <= N - 2:
                        is_peak = nums[i] > nums[i - 1] and nums[i] > nums[i + 1]
                        if is_peak and peaks[i] != 1:
                            fenwick.update(i, 1)
                            peaks[i] = 1
                        elif not is_peak and peaks[i] == 1:
                            fenwick.update(i, -1)
                            peaks[i] = 0
        return answer