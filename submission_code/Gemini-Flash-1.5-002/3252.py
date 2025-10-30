class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i, n):
                temp = nums[:i] + nums[j+1:]
                is_increasing = True
                for k in range(len(temp) - 1):
                    if temp[k] >= temp[k+1]:
                        is_increasing = False
                        break
                if is_increasing:
                    count += 1
        return count