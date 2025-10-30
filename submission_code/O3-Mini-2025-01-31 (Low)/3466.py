from typing import List
from collections import defaultdict

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        # We'll break the array into contiguous segments of "valid" numbers.
        # A number is "valid" if it does not “destroy” any bit that must be present in k.
        # In other words, for every 1-bit in k, the number must also have that bit.
        # This is equivalent to: (num & k) == k.
        #
        # Then for each valid segment we count how many subarrays have bitwise AND == k.
        # We do that using a DP/dictionary approach that keeps track of all
        # bitwise AND values for subarrays ending at the current index.
        #
        # Explanation of the DP approach:
        #   Let dp be a dictionary mapping value -> count for subarrays ending at the previous index.
        #   For the new element x, every new subarray ending at the current index is either
        #       just [x]  or
        #       a previous subarray ending at i-1 extended with x.
        #   So the new dp for index i is:
        #       new_dp[val & x] += count for each val in dp, and also new_dp[x] gets +1.
        #   Then we add new_dp[k] (if exists) to our answer.
        #
        # Note: Outside of a valid segment any subarray that crosses an invalid element
        # cannot yield an AND equal to k (because an invalid element drops a 1-bit needed by k)
        
        res = 0
        n = len(nums)
        i = 0
        
        while i < n:
            # if current element is not valid, skip it.
            if (nums[i] & k) != k:
                i += 1
                continue
            # Otherwise, start a valid segment from index i.
            j = i
            segment = []
            while j < n and ((nums[j] & k) == k):
                segment.append(nums[j])
                j += 1
            # Now process the valid segment.
            dp = defaultdict(int)
            for num in segment:
                new_dp = defaultdict(int)
                # Starting new subarray with the current number.
                new_dp[num] += 1
                # Extend all subarrays ending at previous index.
                for val, count in dp.items():
                    new_and = val & num
                    new_dp[new_and] += count
                # Add count of subarrays ending here with AND equal to k.
                res += new_dp[k]
                dp = new_dp
            # Move i to end of segment.
            i = j
        
        return res