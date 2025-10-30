class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        n = len(nums)
        min_sum = float('inf')
        for j in range(1, n-1):
            left_candidates = [nums[i] for i in range(j) if nums[i] < nums[j]]
            right_candidates = [nums[k] for k in range(j+1, n) if nums[k] < nums[j]]
            if left_candidates and right_candidates:
                current_min = min(left_candidates) + nums[j] + min(right_candidates)
                if current_min < min_sum:
                    min_sum = current_min
        return min_sum if min_sum != float('inf') else -1