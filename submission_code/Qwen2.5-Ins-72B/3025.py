from typing import List
import collections

class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        # Count the occurrences of each power of 2 in nums
        count = collections.Counter(nums)
        
        # Convert target to binary and count the required powers of 2
        required = collections.Counter()
        while target:
            required[target & -target] += 1
            target &= target - 1
        
        # Initialize the number of operations
        operations = 0
        
        # Process each power of 2 from smallest to largest
        for i in range(31):
            power = 1 << i
            if power in required:
                if count[power] >= required[power]:
                    count[power] -= required[power]
                else:
                    # We need to break down a larger power of 2
                    for j in range(i + 1, 31):
                        if count[1 << j] > 0:
                            count[1 << j] -= 1
                            operations += j - i
                            count[power] += (1 << (j - i)) - required[power]
                            break
                    else:
                        return -1
            # Carry over the remaining count to the next power of 2
            count[1 << (i + 1)] += count[power] // 2
        
        return operations