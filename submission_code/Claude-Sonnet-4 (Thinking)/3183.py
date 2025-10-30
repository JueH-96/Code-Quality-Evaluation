class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        result = 0
        
        # Check each bit position (0 to 30, since nums[i] < 2^31)
        for bit_pos in range(31):
            count = 0
            
            # Count how many numbers have this bit set
            for num in nums:
                if (num >> bit_pos) & 1:
                    count += 1
            
            # If at least k numbers have this bit set, set it in result
            if count >= k:
                result |= (1 << bit_pos)
        
        return result