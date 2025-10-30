class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        count = 0
        n = len(nums)
        
        for start in range(n):
            and_value = nums[start]
            if and_value == k:
                count += 1
            for end in range(start + 1, n):
                and_value &= nums[end]
                if and_value < k:
                    break
                if and_value == k:
                    count += 1
        
        return count