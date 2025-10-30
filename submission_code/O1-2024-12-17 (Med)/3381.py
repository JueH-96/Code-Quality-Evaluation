class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ans = float('inf')
        
        # Check all subarrays using two pointers
        for start in range(n):
            current_or = 0
            for end in range(start, n):
                current_or |= nums[end]
                # If the OR so far is >= k, update answer and break
                if current_or >= k:
                    ans = min(ans, end - start + 1)
                    break
        
        return ans if ans != float('inf') else -1