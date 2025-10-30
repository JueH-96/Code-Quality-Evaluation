from typing import List

class Solution:
    def countSubarrays(self, nums: List[int]) -> int:
        """
        Count the number of contiguous subarrays of length 3
        where the sum of the first and third elements equals
        exactly half of the second element.
        """
        n = len(nums)
        count = 0
        
        # Iterate over all windows of size 3
        for i in range(n - 2):
            a, b, c = nums[i], nums[i+1], nums[i+2]
            # Check if sum of first and third is half of the second
            # i.e., 2 * (a + c) == b
            if 2 * (a + c) == b:
                count += 1
        
        return count

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.countSubarrays([1,2,1,4,1]))  # Expected output: 1
    print(sol.countSubarrays([1,1,1]))      # Expected output: 0