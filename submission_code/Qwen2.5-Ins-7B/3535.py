from typing import List

class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        count = 0
        
        for i in range(n):
            for j in range(i, n):
                arr1 = [0] * n
                arr2 = [0] * n
                arr1[i] = nums[i]
                arr2[j] = nums[j]
                for k in range(n):
                    if k == i:
                        continue
                    if k == j:
                        continue
                    if arr1[k] + arr2[k] == nums[k]:
                        arr1[k] = nums[k] - arr2[k]
                        arr2[k] = nums[k] - arr1[k]
                    else:
                        break
                else:
                    count += 1
        
        return count % MOD