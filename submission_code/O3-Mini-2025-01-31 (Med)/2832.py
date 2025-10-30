from typing import List
from collections import defaultdict

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        # Map each number to its list of indices in the array.
        positions = defaultdict(list)
        for i, num in enumerate(nums):
            positions[num].append(i)
        
        max_length = 0
        # For each distinct number, use two pointers on its positions array.
        for pos_list in positions.values():
            l = 0
            # Slide through the indices list using two pointers.
            for r in range(len(pos_list)):
                # Calculate the number of deletions required to make the subarray of the chosen number contiguous.
                # The required deletions = (the gap between pos_list[r] and pos_list[l] + 1) - (number of this element in the window)
                # i.e., pos_list[r] - pos_list[l] + 1 - (r - l + 1) = pos_list[r] - pos_list[l] - (r - l)
                while pos_list[r] - pos_list[l] - (r - l) > k:
                    l += 1
                current_length = r - l + 1
                max_length = max(max_length, current_length)
        return max_length

# Example usage:
if __name__ == "__main__":
    solution = Solution()
    # Test Example 1:
    nums1 = [1, 3, 2, 3, 1, 3]
    k1 = 3
    print(solution.longestEqualSubarray(nums1, k1))  # Expected output: 3
    
    # Test Example 2:
    nums2 = [1, 1, 2, 2, 1, 1]
    k2 = 2
    print(solution.longestEqualSubarray(nums2, k2))  # Expected output: 4