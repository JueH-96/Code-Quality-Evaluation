from typing import List

class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        if n < 3:
            return [0] * len(queries)
        
        peaks = [False] * n
        for j in range(1, n-1):
            peaks[j] = nums[j] > nums[j-1] and nums[j] > nums[j+1]
        
        class SegmentTree:
            def __init__(self, data):
                self.n = len(data)
                self.size = 1
                while self.size < self.n:
                    self.size <<= 1
                self.tree = [0] * (2 * self.size)
                for i in range(self.n):
                    self.tree[self.size + i] = 1 if data[i] else 0
                for i in range(self.size - 1, 0, -1):
                    self.tree[i] = self.tree[2*i] + self.tree[2*i+1]
            
            def update(self, pos, delta):
                pos += self.size
                self.tree[pos] += delta
                pos >>= 1
                while pos >= 1:
                    new_val = self.tree[2*pos] + self.tree[2*pos+1]
                    if self.tree[pos] == new_val:
                        break
                    self.tree[pos] = new_val
                    pos >>= 1
            
            def query_sum(self, l, r):
                res = 0
                l += self.size
                r += self.size
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
        
        st = SegmentTree(peaks)
        answer = []
        
        for q in queries:
            if q[0] == 1:
                l, r = q[1], q[2]
                if r - l < 2:
                    answer.append(0)
                else:
                    a = l + 1
                    b = r - 1
                    if a > b:
                        answer.append(0)
                    else:
                        answer.append(st.query_sum(a, b))
            else:
                index, val = q[1], q[2]
                old_val = nums[index]
                nums[index] = val
                affected = set()
                for dj in [-1, 0, 1]:
                    j = index + dj
                    if 0 < j < n - 1:
                        affected.add(j)
                for j in affected:
                    was_peak = peaks[j]
                    current = nums[j]
                    left = nums[j-1]
                    right = nums[j+1]
                    new_peak = current > left and current > right
                    if new_peak != was_peak:
                        peaks[j] = new_peak
                        delta = 1 if new_peak else -1
                        st.update(j, delta)
        
        return answer