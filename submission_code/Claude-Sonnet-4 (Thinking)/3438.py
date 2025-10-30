class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        def is_peak(arr, i):
            if i <= 0 or i >= len(arr) - 1:
                return False
            return arr[i] > arr[i-1] and arr[i] > arr[i+1]
        
        result = []
        
        for query in queries:
            if query[0] == 1:  # Count peaks in range
                l, r = query[1], query[2]
                count = 0
                for i in range(l+1, r):  # Check indices from l+1 to r-1 inclusive
                    if is_peak(nums, i):
                        count += 1
                result.append(count)
            else:  # Update value
                index, val = query[1], query[2]
                nums[index] = val
        
        return result