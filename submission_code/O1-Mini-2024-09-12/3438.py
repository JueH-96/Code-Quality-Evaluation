from typing import List

class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        class SegmentTree:
            def __init__(self, data):
                n = len(data)
                self.n = 1
                while self.n < n:
                    self.n <<= 1
                self.tree = [0] * (2 * self.n)
                for i in range(n):
                    self.tree[self.n + i] = data[i]
                for i in range(self.n - 1, 0, -1):
                    self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]
            
            def update(self, index, value):
                index += self.n
                self.tree[index] = value
                while index > 1:
                    index >>= 1
                    self.tree[index] = self.tree[2 * index] + self.tree[2 * index + 1]
            
            def query(self, l, r):
                res = 0
                l += self.n
                r += self.n
                while l <= r:
                    if l % 2 == 1:
                        res += self.tree[l]
                        l += 1
                    if r % 2 == 0:
                        res += self.tree[r]
                        r -= 1
                    l >>= 1
                    r >>= 1
                return res

        n = len(nums)
        peak = [0] * n
        for i in range(1, n-1):
            if nums[i] > nums[i-1] and nums[i] > nums[i+1]:
                peak[i] = 1
        st = SegmentTree(peak)
        res = []
        for q in queries:
            if q[0] == 1:
                l, r = q[1], q[2]
                if r - l < 2:
                    res.append(0)
                else:
                    start = l + 1
                    end = r - 1
                    start = max(1, start)
                    end = min(n-2, end)
                    if start > end:
                        res.append(0)
                    else:
                        res.append(st.query(start, end))
            else:
                index, val = q[1], q[2]
                nums[index] = val
                for i in range(index-1, index+2):
                    if 1 <= i <= n-2:
                        new_peak = 1 if nums[i] > nums[i-1] and nums[i] > nums[i+1] else 0
                        if peak[i] != new_peak:
                            peak[i] = new_peak
                            st.update(i, new_peak)
        return res