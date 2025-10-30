class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        for i in range(n):
            for j in range(i, n):
                temp = []
                if i > 0:
                    temp.extend(nums[:i])
                if j < n - 1:
                    temp.extend(nums[j+1:])
                
                if not temp:
                    count += 1
                    continue
                
                is_increasing = True
                for k in range(len(temp) - 1):
                    if temp[k] >= temp[k+1]:
                        is_increasing = False
                        break
                if is_increasing:
                    count += 1
        return count