import collections
from typing import List

class Solution:
    def countNonDecreasingSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        count = 0
        left = 0
        # Deque stores blocks: (sum_of_original_a_values_in_block, count_of_elements_in_block, assigned_b_value_for_block)
        # The assigned_b_value_for_block is determined by the PAVA algorithm for sum(b_i - a_i) minimization
        # subject to b_i >= a_i and b_i non-decreasing. The value for a block is the maximum of the
        # assigned values of the blocks it merges, which comes from the leftmost block in the merge.
        deque = collections.deque()
        current_total_b_sum = 0 # Stores sum(b_val * count) for blocks in deque

        # Precompute prefix sums of original nums for efficient sum(a) calculation
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i+1] = prefix_sum[i] + nums[i]

        for right in range(n):
            current_a = nums[right]

            # When adding nums[right] to the window [left...right]
            # It initially forms a new block (nums[right], 1, nums[right])
            # We merge it with blocks to its left if the non-decreasing property is violated
            new_block_sum_a = current_a
            new_block_count = 1
            new_block_b_val = current_a # Initial potential b value for this element

            # Merge right-to-left: While the new block's b_val is less than the last block's b_val in the deque
            # This condition `new_block_b_val < deque[-1][2]` triggers a merge.
            # The value of the merged block is the value of the left (previous) block (`deque[-1][2]`).
            while deque and new_block_b_val < deque[-1][2]:
                prev_sum_a, prev_count, prev_b_val = deque.pop()
                
                # Remove contribution of the block we are merging from the total b_sum
                current_total_b_sum -= prev_b_val * prev_count
                
                # Merge the previous block into the current new block
                new_block_sum_a += prev_sum_a
                new_block_count += prev_count
                new_block_b_val = prev_b_val # The assigned value for the merged block is the value of the left block

            # Add the final (potentially merged) new block to the deque
            deque.append((new_block_sum_a, new_block_count, new_block_b_val))
            
            # Add the contribution of the final (potentially merged) new block to the total b_sum
            current_total_b_sum += new_block_b_val * new_block_count

            # Now, shrink the window from the left if the cost exceeds k
            # The cost for the window nums[left...right] is current_total_b_sum - sum(nums[left...right])
            # sum(nums[left...right]) = prefix_sum[right+1] - prefix_sum[left]
            
            while left <= right:
                current_sum_a_window = prefix_sum[right+1] - prefix_sum[left]
                current_cost = current_total_b_sum - current_sum_a_window

                if current_cost <= k:
                    # The window [left...right] is valid. All windows [i...right] with left <= i <= right are valid
                    # because the cost is non-increasing as the left endpoint moves right.
                    # The current 'left' is the minimum start index for a valid window ending at 'right'.
                    # So we can break the shrinking loop and add the count of valid start indices.
                    break
                else:
                    # The window [left...right] is invalid (cost > k). We must shrink from the left.
                    # Remove nums[left] from the window.
                    
                    # Get the first block in the deque (corresponds to the leftmost part of the window)
                    first_sum_a, first_count, first_b_val = deque[0]
                    
                    # Remove the contribution of this block from the total b_sum
                    current_total_b_sum -= first_b_val * first_count

                    if first_count == 1:
                        # The first block contains only nums[left]. Remove the block.
                        deque.popleft()
                    else:
                        # The first block contains multiple elements. Remove nums[left] from it.
                        # Update the first block in place in the deque.
                        # sum_a decreases by nums[left], count decreases by 1. b_val remains the same.
                        deque[0] = (first_sum_a - nums[left], first_count - 1, first_b_val)
                    
                    # Move the left pointer
                    left += 1

                    # After removing the first element and updating the deque,
                    # the first block in the potentially updated deque needs its contribution added back.
                    # If the deque is now empty, there's no first block.
                    if deque:
                         new_first_sum_a, new_first_count, new_first_b_val = deque[0]
                         current_total_b_sum += new_first_b_val * new_first_count

            # After the shrinking loop, if left <= right, it means the window [left...right] is valid
            # and 'left' is the minimum starting index for a valid window ending at 'right'.
            # All windows starting from 'left' up to 'right' ending at 'right' are valid.
            # The number of such valid windows is right - left + 1.
            # If the loop finished with left > right, it means even the single element window [right...right]
            # (or whatever the window became) was invalid. In this case, right - left + 1 will be <= 0, correctly adding 0.
            count += (right - left + 1)

        return count