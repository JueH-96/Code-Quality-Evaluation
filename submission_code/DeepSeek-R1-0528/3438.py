class Solution:
    def countOfPeaks(self, nums: list, queries: list) -> list:
        n = len(nums)
        self.n = n
        self.nums = nums
        self.fenw = [0] * (n + 1)
        self.peaks = [0] * n
        
        for i in range(1, n - 1):
            if nums[i] > nums[i - 1] and nums[i] > nums[i + 1]:
                self.peaks[i] = 1
                self._update(i, 1)
        
        res = []
        for q in queries:
            if q[0] == 1:
                l, r = q[1], q[2]
                if l + 1 <= r - 1:
                    res.append(self._query_range(l + 1, r - 1))
                else:
                    res.append(0)
            else:
                idx = q[1]
                val = q[2]
                if self.nums[idx] == val:
                    continue
                self.nums[idx] = val
                for j in [idx - 1, idx, idx + 1]:
                    if j < 1 or j > n - 2:
                        continue
                    new_peak = 1 if (self.nums[j] > self.nums[j - 1] and self.nums[j] > self.nums[j + 1]) else 0
                    if new_peak != self.peaks[j]:
                        self._update(j, new_peak - self.peaks[j])
                        self.peaks[j] = new_peak
        return res

    def _update(self, i, delta):
        idx = i + 1
        while idx <= self.n:
            self.fenw[idx] += delta
            idx += idx & -idx
    
    def _query(self, i):
        if i < 0:
            return 0
        s = 0
        idx = i + 1
        while idx > 0:
            s += self.fenw[idx]
            idx -= idx & -idx
        return s

    def _query_range(self, l, r):
        return self._query(r) - self._query(l - 1)