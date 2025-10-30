class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        peaks = [0] * n
        for i in range(1, n-1):
            if nums[i-1] < nums[i] > nums[i+1]:
                peaks[i] = 1
        # Using a segment tree to handle range queries and updates
        from collections import defaultdict
        class SegmentTree:
            def __init__(self, data):
                self.n = len(data)
                self.size = 1
                while self.size < self.n:
                    self.size <<= 1
                self.tree = [0] * (2 * self.size)
                for i in range(self.n):
                    self.tree[self.size + i] = data[i]
                for i in range(self.size - 1, 0, -1):
                    self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]
            
            def update(self, idx, val):
                idx += self.size
                self.tree[idx] = val
                idx >>= 1
                while idx >= 1:
                    new_val = self.tree[2 * idx] + self.tree[2 * idx + 1]
                    if self.tree[idx] == new_val:
                        break
                    self.tree[idx] = new_val
                    idx >>= 1
            
            def query(self, l, r):
                l += self.size
                r += self.size
                res = 0
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
        res = []
        for q in queries:
            if q[0] == 1:
                l, r = q[1], q[2]
                if r - l < 2:
                    res.append(0)
                else:
                    res.append(st.query(l+1, r-1))
            else:
                idx, val = q[1], q[2]
                nums[idx] = val
                # Update peaks for idx-1, idx, idx+1
                for i in [idx-1, idx, idx+1]:
                    if 0 < i < n-1:
                        if nums[i-1] < nums[i] > nums[i+1]:
                            st.update(i, 1)
                        else:
                            st.update(i, 0)
        return res