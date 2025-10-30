class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        subarray_count = 0
        start_index = 0
        while start_index < n:
            current_and = nums[start_index]
            end_index = start_index
            while end_index < n:
                if current_and == 0:
                    break
                if end_index + 1 < n:
                    current_and = current_and & nums[end_index + 1]
                    if current_and == 0:
                        end_index += 1
                        break
                else:
                    break
                end_index += 1
            subarray_count += 1
            start_index = end_index + 1
        return subarray_count