class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        max_val = max(nums)
        result = 0
        max_positions = []
        
        for i, num in enumerate(nums):
            if num == max_val:
                max_positions.append(i)
            
            if len(max_positions) >= k:
                # Position of the k-th max element from the end
                kth_position = max_positions[len(max_positions) - k]
                result += kth_position + 1
        
        return result