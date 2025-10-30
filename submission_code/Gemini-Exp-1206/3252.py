class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        for i in range(n):
            for j in range(i, n):
                temp = []
                for k in range(n):
                    if k < i or k > j:
                        temp.append(nums[k])
                
                flag = True
                if len(temp) > 1:
                    for k in range(len(temp) - 1):
                        if temp[k] >= temp[k+1]:
                            flag = False
                            break
                if flag:
                    ans += 1
        return ans