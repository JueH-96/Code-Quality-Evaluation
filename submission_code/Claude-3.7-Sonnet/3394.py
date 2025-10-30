class Solution:
    def minEnd(self, n: int, x: int) -> int:
        if n == 1:
            return x
        
        # Find the MSB of x and identify all unset bits up to the MSB
        msb_position = -1
        for i in range(32):  # Assuming 32-bit integers
            if x & (1 << i):
                msb_position = i
        
        unset_positions = []
        for i in range(msb_position + 1):
            if not (x & (1 << i)):
                unset_positions.append(i)
        
        z = len(unset_positions)
        group_size = 1 << z
        
        # Determine which group the (n-1)th element belongs to and its position within the group
        pos = n - 1  # 0-indexed position
        group, index_within_group = divmod(pos, group_size)
        
        # Compute the result
        result = x
        
        # Set the bits at positions beyond MSB based on the binary representation of group
        for i in range(32):
            if group & (1 << i):
                result |= (1 << (msb_position + i + 1))
        
        # Set the bits of the unset positions of x based on the index within group
        for i in range(z):
            if index_within_group & (1 << i):
                result |= (1 << unset_positions[i])
        
        return result