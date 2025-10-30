class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        n = len(nums)
        memo = {}
        mod = 10**9 + 7
        
        def solve(index, last_arr1_val):
            if index == n:
                return 1
            if (index, last_arr1_val) in memo:
                return memo[(index, last_arr1_val)]
            
            lower_bound = 0
            if index > 0:
                lower_bound = max(last_arr1_val + max(0, nums[index] - nums[index-1]), 0)
            upper_bound = nums[index]
            
            count = 0
            for current_arr1_val in range(lower_bound, upper_bound + 1):
                count = (count + solve(index + 1, current_arr1_val)) % mod
                
            memo[(index, last_arr1_val)] = count
            return count
            
        total_count = 0
        for initial_arr1_val in range(nums[0] + 1):
            total_count = (total_count + solve(1, initial_arr1_val)) % mod
            
        return total_count