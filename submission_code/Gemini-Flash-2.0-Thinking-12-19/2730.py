class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        memo = {}

        def solve(index, remaining_k, current_nums):
            if index == n:
                current_or = 0
                for val in current_nums:
                    current_or |= val
                return current_or
            if remaining_k == 0:
                current_or = 0
                for val in current_nums:
                    current_or |= val
                return current_or
            if (index, remaining_k) in memo:
                return memo[(index, remaining_k)]
            
            max_or = 0
            for ops in range(remaining_k + 1):
                next_nums = list(current_nums)
                next_nums[index] *= (2**ops)
                current_or = solve(index + 1, remaining_k - ops, next_nums)
                max_or = max(max_or, current_or)
            
            memo[(index, remaining_k)] = max_or
            return max_or

        return solve(0, k, nums)