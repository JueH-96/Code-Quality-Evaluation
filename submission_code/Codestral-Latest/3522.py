class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        def is_consecutive_and_sorted(arr):
            sorted_arr = sorted(arr)
            return sorted_arr == list(range(sorted_arr[0], sorted_arr[-1] + 1))

        results = []
        for i in range(len(nums) - k + 1):
            subarray = nums[i:i + k]
            if is_consecutive_and_sorted(subarray):
                results.append(max(subarray))
            else:
                results.append(-1)

        return results