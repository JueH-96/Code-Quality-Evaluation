class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        def is_peak(arr, i):
            # Check if element at index i is a peak
            # First and last elements cannot be peaks
            if i == 0 or i == len(arr) - 1:
                return False
            return arr[i] > arr[i-1] and arr[i] > arr[i+1]
        
        result = []
        
        for query in queries:
            if query[0] == 1:  # Count peaks query
                l, r = query[1], query[2]
                count = 0
                
                # Count peaks in the subarray nums[l:r+1]
                # Note: peaks are relative to the original array, not the subarray
                for i in range(l, r + 1):
                    if is_peak(nums, i):
                        count += 1
                
                result.append(count)
                
            else:  # Update query (query[0] == 2)
                index, val = query[1], query[2]
                nums[index] = val
        
        return result