from typing import List

class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        # We can think of the allowed merge operation as “folding” a number into its right neighbor
        # when the left element is less than or equal to the right. Because merges cannot
        # change the order of the remaining elements, an optimal strategy is to try to
        # “absorb” as many numbers from the left into their right neighbors as possible.
        #
        # A very clean solution is to process the array from right to left. Let current
        # denote the sum of a segment that can be merged into one number. Initialize current
        # to the last element. Then for each element nums[i] as we move leftwards, if we can
        # merge it into the current segment (i.e. if nums[i] <= current), then update current = nums[i] + current.
        # Otherwise, we “seal off” the current segment (update our answer if needed)
        # and start a new segment with nums[i]. Finally, the answer is the maximum among
        # all such segments.
        #
        # This works because the merging rule “nums[i] <= nums[i+1]” translates into the condition
        # “an element can be merged with the segment on its right if it is no larger than the merged value.”
        current = nums[-1]
        max_val = current
        
        for i in range(len(nums) - 2, -1, -1):
            if nums[i] <= current:
                # We can merge nums[i] into the current segment.
                current += nums[i]
            else:
                # Cannot merge; finalize the current segment.
                max_val = max(max_val, current)
                # Start a new segment using nums[i].
                current = nums[i]
                
        # Check the last segment.
        max_val = max(max_val, current)
        return max_val

# Example usage and testing:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    nums1 = [2, 3, 7, 9, 3]
    print(sol.maxArrayValue(nums1))  # Expected output: 21

    # Example 2:
    nums2 = [5, 3, 3]
    print(sol.maxArrayValue(nums2))  # Expected output: 11