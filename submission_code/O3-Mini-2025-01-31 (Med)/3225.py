from typing import List

class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        freq = {}
        left = 0
        max_len = 0
        for right, num in enumerate(nums):
            freq[num] = freq.get(num, 0) + 1
            # Shrink the window if the count of the current number exceeds k.
            while freq[num] > k:
                freq[nums[left]] -= 1
                left += 1
            # Update the maximum length found so far.
            max_len = max(max_len, right - left + 1)
        return max_len

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Test Example 1
    print(sol.maxSubarrayLength([1,2,3,1,2,3,1,2], 2))  # Expected output: 6
    # Test Example 2
    print(sol.maxSubarrayLength([1,2,1,2,1,2,1,2], 1))  # Expected output: 2
    # Test Example 3
    print(sol.maxSubarrayLength([5,5,5,5,5,5,5], 4))    # Expected output: 4