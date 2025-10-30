class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        from collections import Counter
        
        # Count the frequency of each power of 2 in nums
        count = Counter(nums)
        operations = 0
        bits = [0] * 31
        
        # Convert target to binary representation
        for i in range(30, -1, -1):
            bits[i] = target & 1
            target >>= 1
        
        # Check if we can form the target without any operations
        if sum(count[i] for i in range(1, 31) if bits[i-1] == 1) >= sum(bits):
            return 0
        
        # If we can't form the target, we need to perform operations
        for i in range(30, -1, -1):
            if bits[i] == 1:
                if count[2**i] > 0:
                    count[2**i] -= 1
                else:
                    j = i
                    while j > 0 and bits[j-1] == 0:
                        j -= 1
                    if j == 0:
                        return -1
                    operations += j
                    for k in range(j, i):
                        count[2**k] += 1
                    count[2**i] = 1
                    bits[j-1] = 0
        
        return operations