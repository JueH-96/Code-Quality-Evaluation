from typing import List

class Solution:
    """
    Solves the problem of finding the sum of imbalance numbers of all subarrays.
    """
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        """
        Calculates the sum of imbalance numbers for all subarrays of nums using an O(n^2) approach.

        The imbalance number of an array arr is defined based on its sorted unique elements (sarr).
        It's the count of indices k such that 0 <= k < len(sarr) - 1 and sarr[k+1] - sarr[k] > 1.

        Args:
            nums: A 0-indexed list of integers. Constraints: 1 <= nums.length <= 1000,
                  and 1 <= nums[i] <= nums.length.

        Returns:
            The sum of imbalance numbers over all contiguous non-empty subarrays of nums.
        """
        n = len(nums)
        # total_imbalance_sum will accumulate the imbalance numbers of all subarrays.
        total_imbalance_sum = 0

        # Iterate through all possible subarrays. A subarray is defined by its start and end indices (i, j).
        # We can iterate through all possible right endpoints j first.
        for j in range(n):
            # For a fixed right endpoint j, we consider all subarrays ending at j: nums[i:j+1] where i ranges from j down to 0.
            
            # As we decrease the left endpoint 'i', we incrementally build the subarray and update its imbalance number.
            # 'seen_elements' stores the unique elements encountered in the subarray nums[i+1:j+1].
            seen_elements = set() 
            # 'imbalance_of_current_subarray' stores the imbalance number of the subarray nums[i+1:j+1].
            # It's updated in each step of the inner loop to reflect the imbalance of nums[i:j+1].
            imbalance_of_current_subarray = 0 
            
            # Iterate through the left endpoint 'i' from j down to 0.
            for i in range(j, -1, -1):
                # Consider the element nums[i], which is being added to the left of the previous subarray (nums[i+1:j+1])
                # to form the current subarray nums[i:j+1].
                current_num = nums[i]

                # We need to calculate how adding 'current_num' changes the imbalance number.
                # The change depends on whether 'current_num' is a new unique element and
                # how it relates to the elements already in 'seen_elements'.
                
                if current_num in seen_elements:
                    # If current_num was already present in the unique elements of nums[i+1:j+1],
                    # adding it again (at index i) does not change the set of unique elements.
                    # Therefore, the imbalance number of nums[i:j+1] is the same as that of nums[i+1:j+1].
                    # 'imbalance_of_current_subarray' remains unchanged.
                    pass 
                else:
                    # If current_num is a new unique element being introduced to the set.
                    
                    # Check if its neighbors (current_num - 1) and (current_num + 1) were present
                    # in the set of unique elements of the previous subarray (nums[i+1:j+1]).
                    has_left_neighbor = (current_num - 1) in seen_elements
                    has_right_neighbor = (current_num + 1) in seen_elements

                    # Update the imbalance based on the presence of these neighbors.
                    # The logic is based on how adding 'current_num' affects the gaps between sorted unique elements.
                    if has_left_neighbor and has_right_neighbor:
                        # Case 1: Both neighbors exist. Adding 'current_num' bridges the gap
                        # between 'current_num - 1' and 'current_num + 1'.
                        # Assuming these were adjacent in the sorted unique set, the gap was (current_num + 1) - (current_num - 1) = 2 > 1.
                        # This gap contributed +1 to the imbalance.
                        # Adding 'current_num' replaces this gap with two gaps of difference 1:
                        # (current_num - (current_num - 1) = 1) and ((current_num + 1) - current_num = 1).
                        # These new gaps do not contribute to imbalance.
                        # The net change in imbalance is -1.
                        imbalance_of_current_subarray -= 1
                    elif not has_left_neighbor and not has_right_neighbor:
                        # Case 2: Neither neighbor exists. 'current_num' is "isolated" from the existing numbers.
                        # Adding an isolated number increases the number of adjacent pairs in the sorted unique list by 1.
                        # If the original set 'seen_elements' was not empty, this new pair (involving current_num and either its
                        # predecessor or successor in the sorted list) will have a difference > 1, increasing the imbalance by 1.
                        # If 'seen_elements' was empty, adding the first element results in 0 imbalance.
                        if seen_elements: # Check if the previous subarray nums[i+1:j+1] had unique elements
                           imbalance_of_current_subarray += 1
                    # Case 3: Exactly one neighbor exists (either left or right).
                    # Adding 'current_num' extends a contiguous block of numbers by 1.
                    # Example: Adding 4 to {..., 3} creates the adjacent pair (3, 4) with difference 1.
                    # Example: Adding 2 to {..., 3} creates the adjacent pair (2, 3) with difference 1.
                    # In both scenarios, the new adjacent pair has a difference of 1, which does not contribute to imbalance.
                    # The overall imbalance count remains unchanged.
                    # else: No change required for imbalance_of_current_subarray

                    # Since 'current_num' was a new unique element, add it to the set 'seen_elements'.
                    # This set will represent the unique elements of nums[i:j+1] for the next iteration (i-1).
                    seen_elements.add(current_num)

                # At this point, 'imbalance_of_current_subarray' holds the correct imbalance number
                # for the subarray nums[i:j+1].
                # Add this value to the total sum accumulating over all subarrays processed so far.
                total_imbalance_sum += imbalance_of_current_subarray

        # After iterating through all possible subarrays (i, j), return the total accumulated sum.
        return total_imbalance_sum