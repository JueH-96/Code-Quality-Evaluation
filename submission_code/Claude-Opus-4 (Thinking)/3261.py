class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        def canAchieve(target):
            # Check if we can achieve OR <= target with at most k operations
            mask = target ^ ((1 << 30) - 1)  # Bits that should not be in the result
            segments = 0
            current_and = (1 << 30) - 1  # All bits set
            
            for num in nums:
                current_and &= num
                if (current_and & mask) == 0:
                    # We can form a valid segment ending here
                    segments += 1
                    current_and = (1 << 30) - 1
            
            # We need n - segments operations to reduce to segments elements
            return n - segments <= k
        
        # Greedy approach: try to turn off each bit from MSB to LSB
        ans = (1 << 30) - 1
        for bit in range(29, -1, -1):
            # Try to turn off this bit
            if canAchieve(ans - (1 << bit)):
                ans -= (1 << bit)
        
        return ans