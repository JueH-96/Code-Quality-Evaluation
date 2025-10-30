class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        # Check if total sum is less than target
        if sum(nums) < target:
            return -1
        
        # Count frequency of each power of 2
        freq = {}
        for num in nums:
            freq[num] = freq.get(num, 0) + 1
        
        operations = 0
        
        # Process each bit of target from least significant to most significant
        bit_position = 0
        while target > 0 or any(freq.values()):
            # Current bit value (power of 2)
            bit_value = 1 << bit_position
            
            # Check if current bit is set in target
            if target & 1:
                if bit_value in freq and freq[bit_value] > 0:
                    # Use one occurrence of this bit value
                    freq[bit_value] -= 1
                else:
                    # Need to split a larger power of 2
                    found = False
                    split_position = bit_position + 1
                    
                    while split_position <= 30:
                        split_value = 1 << split_position
                        if split_value in freq and freq[split_value] > 0:
                            # Found a larger power to split
                            found = True
                            freq[split_value] -= 1
                            
                            # Split down to the required bit position
                            for i in range(split_position - 1, bit_position - 1, -1):
                                operations += 1
                                val = 1 << i
                                freq[val] = freq.get(val, 0) + 1
                            
                            break
                        split_position += 1
                    
                    if not found:
                        return -1
            
            # Combine pairs of current bit value into next bit value
            if bit_value in freq and freq[bit_value] >= 2:
                pairs = freq[bit_value] // 2
                freq[bit_value] %= 2
                next_bit = bit_value * 2
                freq[next_bit] = freq.get(next_bit, 0) + pairs
            
            # Clean up zero frequencies
            if bit_value in freq and freq[bit_value] == 0:
                del freq[bit_value]
            
            target >>= 1
            bit_position += 1
            
            # Stop if we've processed all bits and no more values to combine
            if target == 0 and bit_position > 30:
                break
        
        return operations