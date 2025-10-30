from typing import List

class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        # Count the number of descents in cyclic manner
        count = 0
        pos = -1  # position where the descent occurs
        for i in range(n):
            if nums[i] > nums[(i+1)%n]:
                count += 1
                pos = i
        # If the array is already sorted.
        if count == 0:
            return 0
        # If there are more than one descents, it's not possible.
        if count > 1:
            return -1
        # Otherwise, the array can be rotated to be sorted. The minimum shifts required 
        # is the number needed to bring the smallest element (at pos+1) to index 0.
        # Since after a right shift, element at index i moves to (i+k) mod n, we need:
        # (pos+1+k) mod n == 0 => k = n - (pos+1)
        return n - (pos + 1)
        
# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.minimumRightShifts([3,4,5,1,2]))  # Output: 2
    # Example 2:
    print(sol.minimumRightShifts([1,3,5]))      # Output: 0
    # Example 3:
    print(sol.minimumRightShifts([2,1,4]))      # Output: -1