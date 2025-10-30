class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        result = 0
        
        # Check each bit position (0 to 30 since nums[i] < 2^31)
        for bit_pos in range(31):
            count = 0
            bit_mask = 1 << bit_pos
            
            # Count how many numbers have this bit set
            for num in nums:
                if num & bit_mask:
                    count += 1
            
            # If at least k numbers have this bit set, set it in result
            if count >= k:
                result |= bit_mask
        
        return result