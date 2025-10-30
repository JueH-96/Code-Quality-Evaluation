from typing import List

class Solution:
    def countWays(self, nums: List[int]) -> int:
        # Sort the numbers for binary grounded count determination.
        nums.sort()
        n = len(nums)
        ways = 0
        
        # Helper: count of elements less than x in sorted array = the index of first element >= x.
        # We can do this using binary search. But we'll do a manual pointer iteration since k 
        # will go from 0 to n and we can scan over sorted array efficiently.
        index = 0
        
        # Pre-calculate prefix counts: for each possible k we want the count of numbers < k
        # However, we can simply walk through sorted array.
        # For each candidate selection count k, the valid assignment is forced:
        # 1. All students with nums[i] < k must be selected.
        # 2. All students with nums[i] > k must be not selected.
        # 3. There MUST be exactly k students with nums[i] < k.
        # Also note that a student with nums[i] == k fails both conditions:
        #  - if selected then condition is k > k (false) 
        #  - if not selected then condition is k < k (false)
        # So there must be no student with nums[i] == k.
        
        # To iterate efficiently we can compute for each k the count of numbers that are < k.
        # We use a pointer since nums is sorted.
        j = 0  # pointer into nums
        for k in range(n+1):
            # Move pointer j so that j equals the number of elements that are < k.
            while j < n and nums[j] < k:
                j += 1
                
            # j is now the count of numbers less than k.
            if j != k:
                continue  # condition "count of numbers < k" must be exactly k.
            
            # Now we need to ensure there's no student with nums[i] equal to k.
            # Check the next element in nums, if exists, must be > k.
            if k < n and nums[k] == k:
                continue  # Not valid because that student would have nums[i] == k.
            
            # If we reach here, k is a valid selection count.
            ways += 1
            
        return ways

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1
    print(sol.countWays([1,1]))  # Expected output: 2
    # Example 2
    print(sol.countWays([6,0,3,3,6,7,2,7]))  # Expected output: 3