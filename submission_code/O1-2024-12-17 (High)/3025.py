class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        # Quick check: if total sum of nums is less than target, it's impossible
        if sum(nums) < target:
            return -1
        
        # Count how many times each power of two appears
        # We'll use indices 0..31 (though target bit can go up to 30 since target < 2^31)
        # but define length 32 to safely handle carry from the 30th bit.
        bit_count = [0]*32
        for x in nums:
            # x is guaranteed to be a power of two
            # find which power it is: if x == 2^k, then k = x.bit_length() - 1
            k = x.bit_length() - 1
            bit_count[k] += 1
        
        operations = 0
        
        # We'll iterate through bits 0 to 30 of the target
        # (the 31st bit would be 0 since target < 2^31, but we'll allow i up to 30 inclusive in the loop)
        for i in range(31):
            # Check if the i-th bit of target is set
            need = (target >> i) & 1
            
            # If we don't have enough 2^i to meet the i-th bit need, try to break down bigger powers
            if bit_count[i] < need:
                # We need 1 more (since need is 0 or 1) if bit_count[i] == 0 and need == 1
                j = i + 1
                while j < 32 and bit_count[j] == 0:
                    j += 1
                if j == 32:
                    # No bigger power available to break down
                    return -1
                # Break down 2^j into 2^(j-i) copies of 2^i
                # This requires (j-i) operations
                operations += (j - i)
                bit_count[j] -= 1
                bit_count[i] += (1 << (j - i))
            
            # Now we have enough 2^i to fulfill the i-th bit if needed
            bit_count[i] -= need
            leftover = bit_count[i]
            
            # Combine leftover 2^i in pairs to form 2^(i+1)
            if i < 31:  # if i=30, we add to bit_count[31], which is still in range
                bit_count[i+1] += leftover // 2
            bit_count[i] = leftover % 2
        
        return operations