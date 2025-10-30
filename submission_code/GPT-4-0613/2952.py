class Solution:
    def minimumTime(self, nums1, nums2, x):
        n = len(nums1)
        if sum(nums1) <= x:
            return 0
        prefix = [0]*(n+1)
        suffix = [0]*(n+1)
        for i in range(n):
            prefix[i+1] = prefix[i] + nums1[i]
        for i in range(n-1, -1, -1):
            suffix[i] = suffix[i+1] + nums1[i]
        j = n
        res = float('inf')
        for i in range(n+1):
            if prefix[i] > x:
                break
            while prefix[i] + suffix[j] > x:
                j -= 1
            res = min(res, max(i, n-j))
        return -1 if res == float('inf') else res