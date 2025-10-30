class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        class SegmentTree:
            def __init__(self, data):
                n = len(data)
                self.N = 1
                while self.N < n:
                    self.N <<=1
                self.tree = [0]*(2*self.N)
                for i in range(n):
                    self.tree[self.N+i] = data[i]
                for i in range(self.N - 1, 0, -1):
                    self.tree[i] = self.tree[2*i] + self.tree[2*i+1]

            def update(self, idx, val):
                idx += self.N
                self.tree[idx] = val
                while idx > 1:
                    idx >>=1
                    self.tree[idx] = self.tree[2*idx] + self.tree[2*idx+1]

            def query(self, l, r):
                l += self.N
                r += self.N + 1
                res = 0
                while l < r:
                    if l%2:
                        res += self.tree[l]
                        l +=1
                    if r%2:
                        r -=1
                        res += self.tree[r]
                    l >>=1
                    r >>=1
                return res

        n = len(nums)
        nums = nums[:]
        peaks = [0]*n
        for i in range(1,n-1):
            if nums[i] > nums[i-1] and nums[i] > nums[i+1]:
                peaks[i]=1
        tree = SegmentTree(peaks)
        res = []
        for q in queries:
            if q[0]==1:
                l = max(1, q[1])
                r = min(n-2, q[2])
                if l > r:
                    res.append(0)
                else:
                    total = tree.query(l,r)
                    res.append(total)
            elif q[0]==2:
                index = q[1]
                val = q[2]
                nums[index] = val
                for i in [index-1,index,index+1]:
                    if 1 <= i <= n-2:
                        prev_peak = peaks[i]
                        if nums[i] > nums[i-1] and nums[i] > nums[i+1]:
                            peaks[i] =1
                        else:
                            peaks[i]=0
                        if peaks[i] != prev_peak:
                            tree.update(i, peaks[i])
        return res