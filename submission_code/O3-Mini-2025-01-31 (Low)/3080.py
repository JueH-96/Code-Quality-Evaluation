from typing import List

class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        # Compute the overall bitwise AND; this is the target score.
        overall_and = nums[0]
        for num in nums[1:]:
            overall_and &= num

        count = 0
        # Use a mask that has all 1's for enough bits.
        current = (1 << 31) - 1  # mask with 31 ones is safe because nums[i] <= 10^6
        for num in nums:
            current &= num
            if current == overall_and:
                count += 1
                current = (1 << 31) - 1  # reset for next subarray
                
        # If count is zero then we can only take the entire array as one subarray.
        return count if count > 0 else 1

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.maxSubarrays([1, 0, 2, 0, 1, 2]))  # Expected output: 3
    print(sol.maxSubarrays([5, 7, 1, 3]))          # Expected output: 1