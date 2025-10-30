class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        peaks = [0] * n
        for i in range(1, n-1):
            if nums[i] > nums[i-1] and nums[i] > nums[i+1]:
                peaks[i] = 1
        prefix_sum = [0] * (n+1)
        for i in range(1, n+1):
            prefix_sum[i] = prefix_sum[i-1] + peaks[i-1]
        
        res = []
        for q in queries:
            if q[0] == 1:
                l, r = q[1], q[2]
                count = prefix_sum[r+1] - prefix_sum[l]
                res.append(count)
            else:
                idx, val = q[1], q[2]
                if idx > 0 and nums[idx] > nums[idx-1] and nums[idx] > nums[idx+1]:
                    res[-1] -= 1
                if idx < n-1 and nums[idx] > nums[idx-1] and nums[idx] > nums[idx+1]:
                    res[-1] += 1
                nums[idx] = val
        return res