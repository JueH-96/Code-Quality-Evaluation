class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        result = 0
        
        # Check each bit position from 0 to 30
        for bit in range(31):
            # Count how many numbers have this bit set
            count = 0
            for num in nums:
                # Check if current bit is set in this number
                if num & (1 << bit):
                    count += 1
            
            # If at least k numbers have this bit set, set it in result
            if count >= k:
                result |= (1 << bit)
                
        return result