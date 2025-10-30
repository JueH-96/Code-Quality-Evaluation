class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        min_sum = float('inf')
        found = False

        for sub_len in range(l, r + 1):
            for i in range(n - sub_len + 1):
                current_sum = 0
                for j in range(i, i + sub_len):
                    current_sum += nums[j]
                
                if current_sum > 0:
                    min_sum = min(min_sum, current_sum)
                    found = True

        if found:
            return min_sum
        else:
            return -1