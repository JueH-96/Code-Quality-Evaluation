class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        swap1, swap2 = -1, -1
        for i in range(n):
            if nums[i] == 1:
                swap1 = i
            elif nums[i] == n:
                swap2 = i
        if swap1 == 0 and swap2 == n-1:
            return max(swap1, n-swap2-1)
        elif swap1 == 0:
            return swap2
        else:
            return swap1