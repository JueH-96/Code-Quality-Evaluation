class Solution:
    def minimumPairRemoval(self, nums: List[int]) -> int:
        operations = 0
        while True:
            min_pair = float('inf')
            min_pair_idx = -1
            for i in range(len(nums) - 1):
                if nums[i] + nums[i+1] < min_pair:
                    min_pair = nums[i] + nums[i+1]
                    min_pair_idx = i
            if min_pair_idx == -1:
                break
            nums[min_pair_idx] = min_pair
            nums.pop(min_pair_idx + 1)
            operations += 1
        return operations