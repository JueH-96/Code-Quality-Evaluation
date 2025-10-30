class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        n = len(nums)
        ans = float('inf')
        for j in range(1, n - 1):
            left_min = float('inf')
            for i in range(0, j):
                if nums[i] < nums[j]:
                    if nums[i] < left_min:
                        left_min = nums[i]
            if left_min == float('inf'):
                continue
                
            right_min = float('inf')
            for k in range(j + 1, n):
                if nums[k] < nums[j]:
                    if nums[k] < right_min:
                        right_min = nums[k]
            if right_min == float('inf'):
                continue
                
            total = left_min + nums[j] + right_min
            if total < ans:
                ans = total
                
        return ans if ans != float('inf') else -1