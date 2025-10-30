class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        
        # To track how many decrements we need to perform for each index
        needed_decrements = [0] * (n + 1)
        for i in range(1, n + 1):
            needed_decrements[i] = nums[i - 1]
        
        marked = [False] * (n + 1)
        marked_count = 0
        
        for second in range(1, m + 1):
            index = changeIndices[second - 1]
            
            # If we can mark this index
            if needed_decrements[index] == 0 and not marked[index]:
                marked[index] = True
                marked_count += 1
            
            # If we still need to decrement this index
            if not marked[index] and needed_decrements[index] > 0:
                needed_decrements[index] -= 1
            
            # Check if all indices are marked
            if marked_count == n:
                return second
        
        return -1 if marked_count < n else m