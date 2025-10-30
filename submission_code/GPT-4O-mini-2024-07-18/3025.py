class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        from collections import Counter
        
        # Count the occurrences of each power of 2 in nums
        count = Counter(nums)
        
        # This will hold the number of operations needed
        operations = 0
        
        # We will try to form the target from the largest power of 2 down to the smallest
        for power in sorted(count.keys(), reverse=True):
            while target >= power and count[power] > 0:
                # If we can use this power to contribute to the target
                target -= power
                count[power] -= 1
            
            # If we still need to reach the target and we can split this power
            while target >= power and count[power] == 0:
                # We need to perform an operation to split this power
                operations += 1
                # Each split gives us two of the next lower power
                count[power // 2] += 2
        
        # If we have reduced target to 0, we succeeded
        return operations if target == 0 else -1