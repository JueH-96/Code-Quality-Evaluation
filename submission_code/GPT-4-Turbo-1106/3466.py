class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        count = 0
        for i in range(len(nums)):
            and_product = nums[i]
            if and_product == k:
                count += 1
            for j in range(i+1, len(nums)):
                and_product &= nums[j]
                if and_product == k:
                    count += 1
                if and_product < k:
                    break
        return count