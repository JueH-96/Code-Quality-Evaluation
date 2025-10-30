class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Find the maximum number of bits we need to consider
        max_val = max(nums)
        max_bits = max_val.bit_length()
        
        # Binary search on the answer
        # We try to find the minimum OR value achievable
        result = 0
        
        # Check each bit position from most significant to least
        for bit in range(max_bits - 1, -1, -1):
            # Try to see if we can achieve a result without this bit set
            target = result | ((1 << bit) - 1)
            
            # Check if we can achieve this target with at most k operations
            operations = 0
            current_and = (1 << 30) - 1  # All bits set initially
            
            for num in nums:
                current_and &= num
                
                # If current_and has bits outside our target, we need to continue
                if current_and & ~target:
                    operations += 1
                else:
                    # We can stop here and start fresh
                    current_and = (1 << 30) - 1
            
            # If we couldn't achieve the target with k operations,
            # we must include this bit in our result
            if operations > k:
                result |= (1 << bit)
        
        return result