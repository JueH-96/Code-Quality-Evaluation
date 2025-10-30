from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        # This dictionary holds the AND values for all subarrays ending at the previous index,
        # mapping the computed AND value to the count of subarrays with that value.
        dp = {}
        count = 0
        
        # Iterate through each number in nums.
        for num in nums:
            # For the current number, start a new dictionary.
            new_dp = {}
            # The subarray that consists of only the current number has AND equal to num.
            new_dp[num] = new_dp.get(num, 0) + 1
            
            # For every subarray ending at the previous index, extend the subarray by including num.
            # The new AND value is the AND of the previous value and the current num.
            for val, freq in dp.items():
                new_val = val & num
                new_dp[new_val] = new_dp.get(new_val, 0) + freq
            
            # If any of the subarrays ending at current index have the AND equal to k, add their counts.
            if k in new_dp:
                count += new_dp[k]
            
            # Update dp to the new dictionary representing subarrays ending at the current index.
            dp = new_dp
        
        return count

# Test examples
if __name__ == "__main__":
    sol = Solution()
    print(sol.countSubarrays([1, 1, 1], 1))  # Output: 6
    print(sol.countSubarrays([1, 1, 2], 1))  # Output: 3
    print(sol.countSubarrays([1, 2, 3], 2))  # Output: 2