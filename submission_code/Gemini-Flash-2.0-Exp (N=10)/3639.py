class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        diff = [0] * (n + 1)
        for l, r in queries:
            diff[l] += 1
            diff[r + 1] -= 1
        
        freq = [0] * n
        curr = 0
        for i in range(n):
            curr += diff[i]
            freq[i] = curr
        
        for i in range(n):
            if nums[i] < freq[i]:
                return False
            nums[i] -= freq[i]

        for num in nums:
            if num != 0:
                return False
        
        return True