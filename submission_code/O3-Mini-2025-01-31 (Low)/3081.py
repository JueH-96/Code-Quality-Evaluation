from typing import List

class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Since the array is sorted, the most frequent element is either at the beginning or at the end.
        # We can count the frequency of the first element and the last element.
        # But in general the maximum frequency could be in the middle also.
        # For safety, we compute the maximum frequency over the entire array.
        
        max_freq = 1
        current_freq = 1
        for i in range(1, n):
            if nums[i] == nums[i - 1]:
                current_freq += 1
            else:
                if current_freq > max_freq:
                    max_freq = current_freq
                current_freq = 1
        # Final check for the last group
        max_freq = max(max_freq, current_freq)
        
        # The optimal strategy is to pair a smaller element with a larger element.
        # You cannot pair two equal numbers.
        # Notice that one cannot use an element more than once. In an optimal pairing strategy,
        # each pair involves one element from the majority group and another element from a minority group,
        # if the majority count is high.
        #
        # The maximum number of pairs you can form is limited by:
        #   1. n // 2 (because each pair uses two elements)
        #   2. n - max_freq (because every pair must have a number from the non-majority elements)
        #
        pairs = min(n // 2, n - max_freq)
        return n - 2 * pairs

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1
    print(sol.minLengthAfterRemovals([1,3,4,9]))  # Output: 0
    # Example 2
    print(sol.minLengthAfterRemovals([2,3,6,9]))  # Output: 0
    # Example 3
    print(sol.minLengthAfterRemovals([1,1,2]))    # Output: 1