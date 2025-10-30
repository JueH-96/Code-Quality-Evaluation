class Solution:
    def countOfPeaks(self, nums, queries):
        n = len(nums)
        peaks = [0] * n
        for i in range(1, n - 1):
            if nums[i] > nums[i - 1] and nums[i] > nums[i + 1]:
                peaks[i] = 1
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + peaks[i]
        res = []
        for q in queries:
            if q[0] == 1:
                l, r = q[1], q[2]
                res.append(prefix[r + 1] - prefix[l])
            else:
                idx, val = q[1], q[2]
                if nums[idx] == val:
                    continue
                nums[idx] = val
                for i in range(max(0, idx - 1), min(n, idx + 2)):
                    if i > 0 and i < n - 1 and nums[i] > nums[i - 1] and nums[i] > nums[i + 1]:
                        if peaks[i] == 0:
                            peaks[i] = 1
                            for j in range(i + 1, n + 1):
                                prefix[j] += 1
                    else:
                        if peaks[i] == 1:
                            peaks[i] = 0
                            for j in range(i + 1, n + 1):
                                prefix[j] -= 1
        return res