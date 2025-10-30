class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        n = len(nums)
        count = [0] * (max(nums) + 1)
        for num in nums:
            count[num] += 1
        
        groups = 0
        for i in range(1, len(count)):
            if count[i] > 0:
                groups += 1
                count[i] -= 1
                while count[i] > 0:
                    if i + 1 < len(count) and count[i + 1] > 0:
                        count[i + 1] -= 1
                    else:
                        groups += 1
                        count[i] -= 1
        
        return groups