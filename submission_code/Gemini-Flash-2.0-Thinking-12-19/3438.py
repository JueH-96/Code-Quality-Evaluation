class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        results = []
        for query in queries:
            if query[0] == 1:
                l_i, r_i = query[1], query[2]
                peak_count = 0
                for i in range(l_i + 1, r_i):
                    if nums[i-1] < nums[i] and nums[i] > nums[i+1]:
                        peak_count += 1
                results.append(peak_count)
            elif query[0] == 2:
                index_i, val_i = query[1], query[2]
                nums[index_i] = val_i
        return results