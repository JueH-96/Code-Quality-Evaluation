from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Find the global maximum in the array.
        x = max(nums)
        # If the overall frequency of x is less than k, no subarray can have at least k occurrences.
        if nums.count(x) < k:
            return 0
        
        # Total number of subarrays.
        total = n * (n + 1) // 2
        
        # Count the subarrays that contain at most (k-1) copies of x.
        count_at_most = 0
        left = 0
        freq = 0
        
        for right in range(n):
            if nums[right] == x:
                freq += 1
            # Shrink the window until the current window has at most (k-1) occurrences of x.
            while freq > (k - 1):
                if nums[left] == x:
                    freq -= 1
                left += 1
            # All subarrays ending at 'right' with start indices in [left, right]
            # have at most (k-1) occurrences of x.
            count_at_most += (right - left + 1)
        
        # Subarrays where x appears at least k times.
        return total - count_at_most

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.countSubarrays([1,3,2,3,3], 2))  # Expected output: 6
    # Example 2:
    print(sol.countSubarrays([1,4,2,1], 3))    # Expected output: 0