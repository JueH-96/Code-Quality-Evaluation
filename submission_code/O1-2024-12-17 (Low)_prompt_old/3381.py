class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        min_length = float('inf')
        
        for start in range(n):
            current_or = 0
            for end in range(start, n):
                current_or |= nums[end]
                
                # If the bitwise OR so far meets or exceeds k, update min_length
                if current_or >= k:
                    min_length = min(min_length, end - start + 1)
                    break
        
        return min_length if min_length != float('inf') else -1