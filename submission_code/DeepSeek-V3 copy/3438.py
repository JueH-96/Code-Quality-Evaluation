class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        peaks = [0] * n
        for i in range(1, n-1):
            if nums[i-1] < nums[i] > nums[i+1]:
                peaks[i] = 1
        # To handle range queries efficiently, we can use a prefix sum array
        prefix = [0] * (n+1)
        for i in range(n):
            prefix[i+1] = prefix[i] + peaks[i]
        result = []
        for query in queries:
            if query[0] == 1:
                l, r = query[1], query[2]
                if r - l < 2:
                    result.append(0)
                else:
                    # Since peaks are only for elements between l and r-1
                    # We need to count peaks from l+1 to r-1
                    count = prefix[r] - prefix[l+1]
                    result.append(count)
            else:
                index, val = query[1], query[2]
                nums[index] = val
                # Update peaks for the modified index and its neighbors
                for i in range(max(1, index-1), min(n-1, index+2)):
                    if nums[i-1] < nums[i] > nums[i+1]:
                        peaks[i] = 1
                    else:
                        peaks[i] = 0
                # Rebuild the prefix sum array
                for i in range(n):
                    prefix[i+1] = prefix[i] + peaks[i]
        return result