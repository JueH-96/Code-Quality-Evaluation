class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        result = []
        
        for query in queries:
            if query[0] == 1:  # Type 1: Count peaks in subarray
                l, r = query[1], query[2]
                count = 0
                for i in range(l + 1, r):  # First and last elements can't be peaks
                    if nums[i] > nums[i-1] and nums[i] > nums[i+1]:
                        count += 1
                result.append(count)
            else:  # Type 2: Update value
                index, val = query[1], query[2]
                nums[index] = val
        
        return result