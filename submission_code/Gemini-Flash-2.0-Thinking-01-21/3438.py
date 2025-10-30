class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        def is_peak(arr, index):
            if index <= 0 or index >= len(arr) - 1:
                return False
            return arr[index] > arr[index - 1] and arr[index] > arr[index + 1]

        results = []
        current_nums = list(nums)  # Create a copy to avoid modifying original nums during queries

        for query in queries:
            if query[0] == 1:
                _, l_i, r_i = query
                subarray = current_nums[l_i:r_i+1]
                peak_count = 0
                for i in range(1, len(subarray) - 1):
                    if is_peak(subarray, i):
                        peak_count += 1
                results.append(peak_count)
            elif query[0] == 2:
                _, index_i, val_i = query
                current_nums[index_i] = val_i
        return results