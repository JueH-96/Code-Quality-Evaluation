class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        count = 0
        prefix_modulo = [0] * (len(nums) + 1)
        for i in range(len(nums)):
            prefix_modulo[i+1] = (prefix_modulo[i] + nums[i] % modulo) % modulo
            if prefix_modulo[i+1] == k:
                count += i + 1
        return count