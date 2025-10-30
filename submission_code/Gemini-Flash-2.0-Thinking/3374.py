class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                is_alternating = True
                if len(subarray) > 1:
                    for k in range(len(subarray) - 1):
                        if subarray[k] == subarray[k+1]:
                            is_alternating = False
                            break
                if is_alternating:
                    count += 1
        return count