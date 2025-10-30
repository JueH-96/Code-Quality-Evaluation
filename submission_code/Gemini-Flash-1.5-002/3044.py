class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        count = 0
        collected = set()
        for i in range(len(nums) -1, -1, -1):
            count += 1
            collected.add(nums[i])
            if all(x in collected for x in range(1,k+1)):
                return count