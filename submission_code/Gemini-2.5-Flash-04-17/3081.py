from typing import List

class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Use two pointers.
        # Pointer i starts at the beginning of the first conceptual half (original indices 0 to n//2 - 1).
        # Pointer j starts at the beginning of the second conceptual half (original indices n//2 to n - 1).
        i = 0
        j = n // 2
        
        # Count the number of pairs that can be removed using the specific strategy.
        pairs = 0
        
        # We try to match the element at nums[i] with the element at nums[j].
        # Pointer i iterates through the first half (up to index n//2 - 1).
        # Pointer j iterates through the second half (starting from index n//2).
        # The loop continues as long as there are elements to consider in both conceptual halves.
        # Since i < n//2 and j >= n//2, if the elements at original indices i and j are still in the array,
        # the element originally at index i will appear before the element originally at index j
        # in the current array. Thus, the condition index_of_i < index_of_j is always satisfied
        # when we pair an element from the first half block with an element from the second half block.
        while i < n // 2 and j < n:
            # If the element pointed to by i (from the first half)
            # is strictly less than the element pointed to by j (from the second half),
            # we can form a valid pair and remove them.
            if nums[i] < nums[j]:
                # We found a valid pair to remove.
                pairs += 1
                # Move to the next available element in the first half block.
                i += 1
                # Move to the next available element in the second half block.
                j += 1
            else: # nums[i] >= nums[j]
                # We cannot pair nums[i] with nums[j] because nums[i] is not strictly less than nums[j].
                # Since nums[j] is the current smallest available element from the second half block,
                # and nums[i] >= nums[j], nums[i] cannot be paired with nums[j] or any element
                # before nums[j] in the second half block (as they are <= nums[j]).
                # To find a partner for nums[i] (or potentially other elements from the first half block),
                # we need to consider strictly larger elements from the second half block, which must appear after nums[j].
                # We advance pointer j to the next element in the second half block.
                # nums[i] remains available to be matched with a later element in the second half block.
                j += 1
        
        # The total number of elements removed by this strategy is 2 * pairs.
        # The minimum length remaining is the original length minus the removed elements.
        # This greedy strategy of maximizing pairs between the first half and the second half
        # is optimal for this problem.
        min_length = n - 2 * pairs
        
        return min_length