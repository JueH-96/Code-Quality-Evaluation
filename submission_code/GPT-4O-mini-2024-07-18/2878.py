class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        current_decrement = 0
        for i in range(n):
            if i >= k:
                current_decrement -= nums[i - k]
            if nums[i] < current_decrement:
                return False
            current_decrement += nums[i] - current_decrement
        return current_decrement == 0