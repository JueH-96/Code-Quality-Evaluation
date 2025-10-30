class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        count = 0
        
        for start in range(n):
            current_and = nums[start]
            if current_and == k:
                count += 1
            for end in range(start + 1, n):
                current_and &= nums[end]
                if current_and == k:
                    count += 1
                elif current_and < k:  # Once AND becomes less than k, it can't ever become k again
                    break
        
        return count