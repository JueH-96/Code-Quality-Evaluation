class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        n = len(nums)
        best = float('inf')
        
        for j in range(1, n - 1):
            for i in range(j):
                if nums[i] < nums[j]:
                    for k in range(j + 1, n):
                        if nums[k] < nums[j]:
                            current_sum = nums[i] + nums[j] + nums[k]
                            if current_sum < best:
                                best = current_sum
        
        return best if best != float('inf') else -1