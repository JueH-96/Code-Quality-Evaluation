class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        count = 0
        for i in range(len(nums)):
            bitwise_and = nums[i]
            for j in range(i, len(nums)):
                bitwise_and &= nums[j]
                if bitwise_and == k:
                    count += 1
                elif bitwise_and < k:
                    break
        return count