class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n, m = len(nums), len(changeIndices)
        marked = [False] * n
        
        for second in range(1, m+1):
            index = changeIndices[second-1] - 1
            if nums[index] == 0:
                marked[index] = True
            else:
                nums[index] -= 1
            
            if all(marked):
                return second
        
        return -1