from typing import List

class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        n = len(nums)
        MOD = 10**9 + 7
        
        dp = {}

        def count_pairs_recursive(index, prev_arr1, prev_arr2):
            if index == n:
                return 1
            
            if (index, prev_arr1, prev_arr2) in dp:
                return dp[(index, prev_arr1, prev_arr2)]
            
            count = 0
            for arr1_val in range(prev_arr1, nums[index] + 1):
                arr2_val = nums[index] - arr1_val
                if arr2_val <= prev_arr2:
                    count = (count + count_pairs_recursive(index + 1, arr1_val, arr2_val)) % MOD
            
            dp[(index, prev_arr1, prev_arr2)] = count
            return count

        return count_pairs_recursive(0, 0, 50)