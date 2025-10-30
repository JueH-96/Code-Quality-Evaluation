class Solution:
    def countPairs(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                num1 = str(nums[i])
                num2 = str(nums[j])
                if len(num1) != len(num2):
                    continue
                diff = 0
                for k in range(len(num1)):
                    if num1[k] != num2[k]:
                        diff += 1
                        if diff > 2:
                            break
                if diff == 0 or diff == 2:
                    count += 1
        return count