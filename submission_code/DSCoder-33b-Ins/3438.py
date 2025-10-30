class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        peaks = [0]*n
        for i in range(1, n-1):
            if nums[i] > nums[i-1] and nums[i] > nums[i+1]:
                peaks[i] = 1
        prefix_sum = [0]*(n+1)
        for i in range(n):
            prefix_sum[i+1] = prefix_sum[i] + peaks[i]
        result = []
        for query in queries:
            if query[0] == 1:
                l, r = query[1], query[2]
                result.append(prefix_sum[r+1] - prefix_sum[l])
            else:
                index, val = query[1], query[2]
                if index > 0 and index < n-1:
                    if nums[index] > nums[index-1] and nums[index] > nums[index+1]:
                        peaks[index] = 0
                    nums[index] = val
                    if index > 0 and index < n-1:
                        if nums[index] > nums[index-1] and nums[index] > nums[index+1]:
                            peaks[index] = 1
                    prefix_sum = [0]*(n+1)
                    for i in range(n):
                        prefix_sum[i+1] = prefix_sum[i] + peaks[i]
        return result