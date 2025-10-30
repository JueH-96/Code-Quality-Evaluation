class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        collected = set()
        operations = 0
        for i in range(len(nums) - 1, -1, -1):
            operations += 1
            collected.add(nums[i])
            if len(collected) == k:
                all_collected = True
                for j in range(1, k + 1):
                    if j not in collected:
                        all_collected = False
                        break
                if all_collected:
                    return operations
        return operations