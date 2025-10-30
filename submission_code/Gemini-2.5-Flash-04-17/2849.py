from typing import List
from sortedcontainers import SortedList

class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        total_sum_imbalance = 0

        # Iterate through all possible start indices i
        for i in range(n):
            # current_sorted_list stores the sorted elements of the subarray nums[i...j]
            # As we extend the subarray from nums[i...i] to nums[i...n-1], we add one element at a time.
            # Using SortedList allows efficient insertion and finding neighbors.
            current_sorted_list = SortedList()
            # current_imbalance stores the imbalance number for the subarray nums[i...j]
            current_imbalance = 0

            # Iterate through all possible end indices j (j >= i)
            # This loop defines the current subarray nums[i...j]
            for j in range(i, n):
                v = nums[j]

                # Find the insertion index for v in the current sorted list (before adding v)
                # bisect_left finds the first index `idx` such that `current_sorted_list[idx]` >= `v`.
                idx = current_sorted_list.bisect_left(v)

                # Calculate change in imbalance caused by adding v
                delta_imbalance = 0

                # The new adjacent pairs involving v in the *new* sorted list (after adding v)
                # will potentially create new imbalance gaps.
                # The elements that become v's neighbors in the *new* sorted list were
                # `current_sorted_list[idx - 1]` and `current_sorted_list[idx]` in the list *before* adding v.

                # Check for new gap created by inserting v on its left side
                # The new adjacent pair is (element_before, v).
                if idx > 0:
                    left_neighbor = current_sorted_list[idx - 1]
                    # A new imbalance gap (left_neighbor, v) is created if v is strictly greater than its left neighbor + 1
                    # If v == left_neighbor or v == left_neighbor + 1, the difference is 0 or 1, no imbalance.
                    if v - left_neighbor > 1:
                        delta_imbalance += 1

                # Check for new gap created by inserting v on its right side
                # The new adjacent pair is (v, element_after).
                # This element `right_neighbor` is at index `idx` in the list *before* adding v.
                if idx < len(current_sorted_list):
                    right_neighbor = current_sorted_list[idx]
                    # A new imbalance gap (v, right_neighbor) is created if right neighbor is strictly greater than v + 1
                    # If v == right_neighbor or v + 1 == right_neighbor, the difference is 0 or 1, no imbalance.
                    if right_neighbor - v > 1:
                        delta_imbalance += 1

                # Check if v closed an existing gap > 1 between its new neighbors (in the list *before* adding v)
                # If idx > 0 and idx < len(current_sorted_list), the elements at idx-1 and idx were adjacent
                # in the list *before* adding v.
                # If the gap between current_sorted_list[idx-1] and current_sorted_list[idx] was > 1,
                # this original adjacent pair's contribution to imbalance is removed because they are no longer adjacent
                # after v is inserted between them.
                if idx > 0 and idx < len(current_sorted_list):
                    left_neighbor = current_sorted_list[idx - 1]
                    right_neighbor = current_sorted_list[idx]
                    if right_neighbor - left_neighbor > 1:
                         # If the two elements that become v's neighbors (and were adjacent before v)
                         # had a gap > 1, that gap contribution is removed.
                        delta_imbalance -= 1

                # Add v to the sorted list
                current_sorted_list.add(v)
                # Update the current imbalance number
                current_imbalance += delta_imbalance
                # Add the imbalance number of the current subarray nums[i...j] to the total sum
                total_sum_imbalance += current_imbalance

        return total_sum_imbalance