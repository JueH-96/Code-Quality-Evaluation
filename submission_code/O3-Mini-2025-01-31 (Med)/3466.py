from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        ans = 0
        # dp will store for the ending at the previous index: 
        # a mapping from AND value to the count of subarrays ending here with that AND.
        prev = {}
        
        for num in nums:
            curr = {num: 1}  # A new subarray starting and ending at current index.
            
            # Extend each subarray that ended at previous index with the current number.
            for val, cnt in prev.items():
                new_val = val & num  # The AND for the extended subarray.
                # Add the count into the current dictionary.
                curr[new_val] = curr.get(new_val, 0) + cnt
                
            # If any subarray ending here has AND equal to k, add that to answer.
            ans += curr.get(k, 0)
            
            # Set current dictionary as previous for the next iteration.
            prev = curr
        
        return ans

# Example Test Cases
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.countSubarrays([1, 1, 1], 1))  # Expected output: 6
    # Example 2:
    print(sol.countSubarrays([1, 1, 2], 1))  # Expected output: 3
    # Example 3:
    print(sol.countSubarrays([1, 2, 3], 2))  # Expected output: 2