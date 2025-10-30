class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        count = 0
        
        for i in range(n):
            for j in range(n):
                arr1 = [0] * n
                arr2 = [0] * n
                arr1[i] = nums[i] - nums[j]
                arr2[j] = nums[j]
                if arr1[i] >= 0 and arr2[j] >= 0 and all(arr1[k] <= arr1[k + 1] for k in range(n - 1)) and all(arr2[k] >= arr2[k + 1] for k in range(n - 1)):
                    count = (count + 1) % MOD
        
        return count