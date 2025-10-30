class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        
        # Helper function to check if index i is a peak
        def is_peak(i):
            if i <= 0 or i >= n - 1:
                return False
            return nums[i] > nums[i-1] and nums[i] > nums[i+1]
        
        # Precompute peaks
        peaks = [False] * n
        for i in range(1, n-1):
            peaks[i] = is_peak(i)
        
        result = []
        
        for query in queries:
            if query[0] == 1:  # Count query
                l, r = query[1], query[2]
                count = 0
                # Count peaks from l+1 to r-1 (excluding first and last of subarray)
                for i in range(l+1, r):
                    if peaks[i]:
                        count += 1
                result.append(count)
            
            else:  # Update query
                index, val = query[1], query[2]
                nums[index] = val
                
                # Update peak status for affected positions
                if index - 1 > 0:
                    peaks[index-1] = is_peak(index-1)
                if 0 < index < n-1:
                    peaks[index] = is_peak(index)
                if index + 1 < n-1:
                    peaks[index+1] = is_peak(index+1)
        
        return result