from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        ans = 0
        # prev holds pairs [and_value, count] for subarrays ending at the previous index
        prev = []
        for num in nums:
            # Start new list for subarrays ending here
            curr = [[num, 1]]
            # Single-element subarray [num]
            if num == k:
                ans += 1
            # Extend all previous subarrays by current num
            for val, count in prev:
                new_val = val & num
                # If the AND result matches the last entry, accumulate counts
                if curr[-1][0] == new_val:
                    curr[-1][1] += count
                else:
                    curr.append([new_val, count])
                # If the AND equals k, add those subarrays to the answer
                if new_val == k:
                    ans += count
            # Move to next iteration
            prev = curr
        return ans