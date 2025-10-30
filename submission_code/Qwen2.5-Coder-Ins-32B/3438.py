from typing import List

class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        def is_peak(i):
            return nums[i] > nums[i - 1] and nums[i] > nums[i + 1]
        
        n = len(nums)
        peaks = [0] * n
        for i in range(1, n - 1):
            if is_peak(i):
                peaks[i] = 1
        
        prefix_peaks = [0] * n
        for i in range(1, n):
            prefix_peaks[i] = prefix_peaks[i - 1] + peaks[i]
        
        answer = []
        for query in queries:
            if query[0] == 1:
                l, r = query[1], query[2]
                count = prefix_peaks[r] - prefix_peaks[l]
                if l > 0 and is_peak(l):
                    count += 1
                if r < n - 1 and is_peak(r):
                    count += 1
                answer.append(count)
            elif query[0] == 2:
                index, val = query[1], query[2]
                if nums[index] == val:
                    continue
                nums[index] = val
                for i in [index - 1, index, index + 1]:
                    if 0 < i < n - 1:
                        peaks[i] = is_peak(i)
                for i in range(index, n):
                    prefix_peaks[i] = prefix_peaks[i - 1] + peaks[i]
        
        return answer