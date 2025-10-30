class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        def is_peak(i):
            if i == 0 or i == len(nums) - 1:
                return False
            return nums[i] > nums[i - 1] and nums[i] > nums[i + 1]
        
        peaks = [i for i in range(1, len(nums) - 1) if is_peak(i)]
        result = []
        
        for query in queries:
            if query[0] == 1:
                l, r = query[1], query[2]
                count = 0
                for i in range(l, r + 1):
                    if is_peak(i):
                        count += 1
                result.append(count)
            elif query[0] == 2:
                index, val = query[1], query[2]
                if is_peak(index):
                    peaks.remove(index)
                nums[index] = val
                if is_peak(index):
                    peaks.append(index)
        
        return result