class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        min_sum = float('inf')
        # Iterate over all possible first split points i
        for i in range(n - 2):
            # Iterate over all possible second split points j after i
            for j in range(i + 1, n - 1):
                current_sum = nums[0] + nums[i + 1] + nums[j + 1]
                if current_sum < min_sum:
                    min_sum = current_sum
        return min_sum