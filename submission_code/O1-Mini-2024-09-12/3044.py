class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        desired = set(range(1, k + 1))
        operations = 0
        for num in reversed(nums):
            operations += 1
            if num in desired:
                desired.remove(num)
                if not desired:
                    break
        return operations