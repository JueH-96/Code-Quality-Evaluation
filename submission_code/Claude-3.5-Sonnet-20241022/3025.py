class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        # Count frequency of each power of 2
        count = [0] * 31
        for num in nums:
            bit = 0
            while num > (1 << bit):
                bit += 1
            count[bit] += 1
            
        operations = 0
        curr_bit = 0
        
        # Process each bit of target
        while curr_bit < 31:
            if target & (1 << curr_bit):
                # Need this bit
                if count[curr_bit] > 0:
                    # Have it directly
                    count[curr_bit] -= 1
                else:
                    # Need to find larger number to break down
                    found = False
                    for j in range(curr_bit + 1, 31):
                        if count[j] > 0:
                            # Found larger number, break it down
                            count[j] -= 1
                            for k in range(j-1, curr_bit-1, -1):
                                count[k] += 2
                            operations += j - curr_bit
                            found = True
                            break
                    if not found:
                        return -1
            
            # Add carry to next bit
            count[curr_bit + 1] += count[curr_bit] // 2
            curr_bit += 1
            
        return operations