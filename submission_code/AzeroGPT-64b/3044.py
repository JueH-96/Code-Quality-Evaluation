class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        ops = 0
        target_set = set(range(1, k+1))
        for i in range(len(nums)-1, -1, -1):
            if nums[i] in target_set:
                target_set.remove(nums[i])
                ops += 1
                if not target_set:
                    break
                
        return ops