class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ans = 0
        for i in range(n):
            for j in range(i, n):
                for x in range(-50, 51):
                    temp_nums = nums[:]
                    for l in range(i, j + 1):
                        temp_nums[l] += x
                    
                    count = 0
                    for num in temp_nums:
                        if num == k:
                            count += 1
                    ans = max(ans, count)
        return ans