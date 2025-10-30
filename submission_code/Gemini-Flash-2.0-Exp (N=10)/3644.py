class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        min_sum = float('inf')
        found = False

        for i in range(n):
            for j in range(i, n):
                sub_len = j - i + 1
                if l <= sub_len <= r:
                    sub_sum = sum(nums[i:j+1])
                    if sub_sum > 0:
                        min_sum = min(min_sum, sub_sum)
                        found = True
        
        if found:
            return min_sum
        else:
            return -1