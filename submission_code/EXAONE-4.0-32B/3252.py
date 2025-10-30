class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        for i in range(n):
            for j in range(i, n):
                last = -1
                valid = True
                for k in range(0, i):
                    if nums[k] > last:
                        last = nums[k]
                    else:
                        valid = False
                        break
                if not valid:
                    continue
                for k in range(j + 1, n):
                    if nums[k] > last:
                        last = nums[k]
                    else:
                        valid = False
                        break
                if valid:
                    count += 1
        return count