class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        def check(mask, k):
            # Check if we can eliminate all bits in mask using at most k operations
            ops = 0
            current = 0
            
            for num in nums:
                # Update current segment's AND value
                current = current & num if current > 0 else num
                
                # If current segment has any bits in mask, we need to start a new segment
                if current & mask:
                    ops += 1
                    current = num
            
            return ops <= k
        
        result = 0
        # Try to eliminate each bit from high to low
        for bit in range(29, -1, -1):
            # If we can't eliminate this bit, include it in result
            if not check(1 << bit, k):
                result |= (1 << bit)
        
        return result