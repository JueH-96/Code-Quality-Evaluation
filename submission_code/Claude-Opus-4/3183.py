class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        result = 0
        
        # Check each bit position (0 to 30)
        for bit in range(31):
            count = 0
            
            # Count how many numbers have this bit set
            for num in nums:
                if (num >> bit) & 1:
                    count += 1
            
            # If at least k numbers have this bit set, include it in result
            if count >= k:
                result |= (1 << bit)
        
        return result