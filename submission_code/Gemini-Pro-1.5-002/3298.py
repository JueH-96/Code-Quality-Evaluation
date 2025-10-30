class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        ans = 0
        for i in range(n):
            temp = []
            for j in range(n):
                if nums[j] >= nums[i] and nums[j] <= nums[i] + 1:
                    temp.append(nums[j])
                elif nums[j] == nums[i] -1:
                    temp.append(nums[j])
            
            count = 0
            temp.sort()
            
            for k in range(len(temp)):
                if k == 0 or temp[k] == temp[k-1] or temp[k] == temp[k-1] + 1:
                    count += 1
                else:
                    ans = max(ans, count)
                    count = 1
            ans = max(ans, count)
        return ans