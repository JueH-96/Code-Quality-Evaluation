class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        
        # We'll use a difference array "diff" to track how many flips begin at each index,
        # so we can maintain how many flips are currently affecting the current index as we move.
        diff = [0] * (n + 3)
        
        flips_active = 0  # how many flips are currently affecting the position
        operations = 0
        
        for i in range(n):
            # Incorporate flips that start or stop at this position
            flips_active += diff[i]
            
            # Determine the effective bit (original bit XOR flips_active%2)
            # If flips_active is odd and the original bit=0 => it's effectively 1, etc.
            effective_bit = nums[i] ^ (flips_active % 2)
            
            # If the effective bit is 0, we need to flip at i (cover i, i+1, i+2)
            if effective_bit == 0:
                if i + 2 >= n:
                    # Not enough room to flip 3 consecutive bits => impossible
                    return -1
                operations += 1
                flips_active += 1  # new flip affects everything from i onwards
                diff[i + 3] -= 1    # flip effect ends after position i+2
        
        # If we reach here, all bits are effectively 1
        return operations