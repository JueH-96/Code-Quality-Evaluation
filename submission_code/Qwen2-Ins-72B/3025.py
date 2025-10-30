class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        if sum(nums) < target:
            return -1
        nums.sort(reverse=True)
        ans = 0
        while target > 0:
            if nums[0] > target:
                nums[0] //= 2
                nums.append(nums[0])
                ans += 1
            else:
                target -= nums[0]
                nums.pop(0)
        return ans