from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        k = 3
        ans = 0
        cur = 0  # current parity of flips affecting current index
        # We'll use an auxiliary list to mark where flips start; its value is 1 when a flip starts at that index.
        isFlipped = [0] * n
        
        # Process each index greedily.
        for i in range(n):
            # Remove the effect of the flip that goes out of the window.
            if i >= k:
                cur ^= isFlipped[i - k]
            # The effective value of nums[i] after accounting for all flips so far.
            # Since flipping is its own inverse (i.e. x^flip_parity), the effective value is:
            #   effective_value = nums[i] XOR cur
            # We want effective_value to be 1.
            if (nums[i] ^ cur) == 0:
                # If there aren’t enough indices remaining for a triple flip, it is impossible.
                if i > n - k:
                    return -1
                # Otherwise, flip starting at index i.
                ans += 1
                isFlipped[i] = 1
                cur ^= 1  # update current parity effect as we just started a new flip
                
        return ans

# Additional code for local testing purposes
if __name__ == "__main__":
    sol = Solution()
    
    # Example 1: Expected output 3
    nums1 = [0, 1, 1, 1, 0, 0]
    print(sol.minOperations(nums1))  # Output: 3

    # Example 2: Expected output -1
    nums2 = [0, 1, 1, 1]
    print(sol.minOperations(nums2))  # Output: -1