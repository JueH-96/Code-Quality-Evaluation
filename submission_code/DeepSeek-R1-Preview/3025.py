class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        total = sum(nums)
        if total < target:
            return -1
        
        # Initialize frequency array for exponents 0 to 30
        freq = [0] * 31
        for num in nums:
            exponent = 0
            n = num
            while n > 1:
                n >>= 1
                exponent += 1
            freq[exponent] += 1
        
        operations = 0
        
        for i in range(31):
            # Check if the i-th bit is set in target
            if (target >> i) & 1:
                if freq[i] > 0:
                    freq[i] -= 1
                else:
                    # Find the smallest j > i where freq[j] > 0
                    j = i + 1
                    while j < 31 and freq[j] == 0:
                        j += 1
                    if j == 31:
                        return -1  # No higher bit available
                    
                    operations += (j - i)
                    # Split the higher bit down to i
                    freq[j] -= 1
                    for k in range(j-1, i, -1):
                        freq[k] += 2
                    # Now, add 2 to i and subtract 1
                    freq[i] += 2
                    freq[i] -= 1
                
                # Carry over any remaining bits after using one
                carry = freq[i] // 2
                freq[i+1] += carry
                freq[i] %= 2
            else:
                # Carry over to the next higher bit
                carry = freq[i] // 2
                freq[i+1] += carry
                freq[i] %= 2
        
        return operations