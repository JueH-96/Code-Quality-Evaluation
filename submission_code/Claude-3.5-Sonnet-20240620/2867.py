class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Find the first and last occurrence of 1
        first_one = -1
        last_one = -1
        for i, num in enumerate(nums):
            if num == 1:
                if first_one == -1:
                    first_one = i
                last_one = i
        
        # If there's no 1 in the array, return 0
        if first_one == -1:
            return 0
        
        # If there's only one 1 in the array, return 1
        if first_one == last_one:
            return 1
        
        result = 1
        prev_one = first_one
        
        for i in range(first_one + 1, last_one + 1):
            if nums[i] == 1:
                # Calculate the number of zeros between two 1s
                zeros = i - prev_one
                # Multiply the result by the number of ways to split these zeros
                result = (result * zeros) % MOD
                prev_one = i
        
        return result