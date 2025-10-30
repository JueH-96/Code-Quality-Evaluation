class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        # Count available powers of 2 using their exponents
        count = [0] * 32
        for num in nums:
            exp = num.bit_length() - 1
            count[exp] += 1
        
        # Check if total sum is at least target
        if sum(nums) < target:
            return -1
        
        operations = 0
        
        # Process each bit of target from least significant to most significant
        for i in range(32):
            # Check if we need 2^i
            if target & (1 << i):
                # We need one 2^i
                if count[i] > 0:
                    count[i] -= 1
                else:
                    # We need to create a 2^i by breaking down a larger power
                    j = i + 1
                    while j < 32 and count[j] == 0:
                        j += 1
                    
                    if j == 32:
                        return -1  # Cannot create 2^i
                    
                    # Break down 2^j to create 2^i
                    count[j] -= 1
                    operations += j - i
                    # This creates 2^(j-i) copies of 2^i
                    count[i] += (1 << (j - i)) - 1  # -1 because we use one
            
            # Combine pairs of 2^i to form 2^(i+1)
            count[i + 1] += count[i] // 2
            count[i] %= 2
        
        return operations