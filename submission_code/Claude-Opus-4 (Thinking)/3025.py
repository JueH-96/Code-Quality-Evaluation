class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        # Check if sum is less than target
        if sum(nums) < target:
            return -1
        
        # Count frequency of each bit position
        freq = [0] * 31
        for num in nums:
            bit = num.bit_length() - 1  # position of the highest set bit
            freq[bit] += 1
        
        operations = 0
        
        for i in range(31):  # process all bits
            if target & (1 << i):  # if bit i is set in target
                if freq[i] > 0:
                    freq[i] -= 1
                else:
                    # Need to split from a larger bit
                    j = i + 1
                    while j < 31 and freq[j] == 0:
                        j += 1
                    
                    if j == 31:
                        # This shouldn't happen if sum >= target
                        return -1
                    
                    # Split down from position j to position i
                    while j > i:
                        freq[j] -= 1
                        freq[j-1] += 2
                        operations += 1
                        j -= 1
                    
                    freq[i] -= 1
            
            # Combine pairs to form the next higher bit
            if i < 30:
                freq[i+1] += freq[i] // 2
        
        return operations