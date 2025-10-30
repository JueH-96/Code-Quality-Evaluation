class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        import math
        
        # Quick check: if total sum of nums is less than target, it's impossible
        s = sum(nums)
        if s < target:
            return -1
        
        # Count how many of each power of 2 we have.
        # Since nums[i] is a power of 2, if nums[i] = 2^k, then k = (nums[i]).bit_length() - 1
        freq = [0]*31
        for x in nums:
            p = x.bit_length() - 1  # power k for 2^k
            freq[p] += 1
        
        ops = 0
        carry = 0  # how many 2^i we still have (leftover pairs) from previous bits
        
        # We'll iterate over bits from 0 to 30 (since target < 2^31)
        for i in range(31):
            # Determine how many 2^i we need this round: bit i of target plus any carry from smaller bits
            needed = carry + ((target >> i) & 1)
            
            # If we don't have enough 2^i in freq[i], we try to split larger powers to get more 2^i
            if freq[i] < needed:
                missing = needed - freq[i]
                # Try to produce "missing" more 2^i from bigger powers
                j = i + 1
                while missing > 0 and j < 31:
                    if freq[j] > 0:
                        # Splitting one 2^j creates 2^(j-i) copies of 2^i
                        freq[j] -= 1
                        freq[i] += (1 << (j - i))
                        ops += (j - i)  # each split from j down to i costs (j - i) operations in total
                        
                        if freq[i] >= needed:
                            missing = 0
                        else:
                            missing = needed - freq[i]
                    else:
                        j += 1
                
                if missing > 0:  # still can't produce enough 2^i
                    return -1
            
            # Now we should have at least 'needed' copies of 2^i
            leftover = freq[i] - needed
            # We "use" exactly 'needed' of them to match the target's bit i
            # leftover are extra 2^i that can pair up to form 2^(i+1)
            
            # Next bit can be formed by leftover // 2
            if i < 30:
                freq[i+1] += leftover // 2
            
            carry = leftover // 2
            # Keep exactly leftover % 2 for freq[i] (though it won't really help for subsequent sums,
            # we store it for completeness)
            freq[i] = leftover % 2
        
        return ops